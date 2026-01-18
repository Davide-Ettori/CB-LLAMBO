import os
import time
import openai
import asyncio
import re
import numpy as np
from scipy.stats import norm
from aiohttp import ClientSession
from llambo.rate_limiter import RateLimiter
from llambo.discriminative_sm_utils import gen_prompt_tempates, gen_prompt_tempates_with_reasoning, build_bandit_prompt


API_TYPE = os.getenv("OPENAI_API_TYPE", "open_ai")
openai.api_type = API_TYPE
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
openai.api_key = os.getenv("OPENAI_API_KEY")


class LLM_DIS_SM:
    def __init__(self, task_context, n_gens, lower_is_better,
                 bootstrapping=False, n_templates=1,
                 use_recalibration=False,
                 rate_limiter=None, warping_transformer=None,
                 verbose=False, chat_engine=None,
                 prompt_setting=None, shuffle_features=False,
                 eval_mode="mc",
                 bandit_strategy="ucb1",
                 bandit_budget_multiplier=3,
                 bandit_top_k=3,
                 conformal_quantile=0.9,
                 embedding_model_name="all-MiniLM-L6-v2",
                 reasoning_max_tokens=64):
        '''Initialize the forward LLM surrogate model. This is modelling p(y|x) as in GP/SMAC etc.'''
        self.task_context = task_context
        self.n_gens = n_gens
        self.lower_is_better = lower_is_better
        self.bootstrapping = bootstrapping
        self.n_templates = n_templates
        assert not (bootstrapping and use_recalibration), 'Cannot do recalibration and boostrapping at the same time' 
        self.use_recalibration = use_recalibration
        if rate_limiter is None:
            self.rate_limiter = RateLimiter(max_tokens=100000, time_frame=60)
        else:
            self.rate_limiter = rate_limiter
        if warping_transformer is not None:
            self.warping_transformer = warping_transformer
            self.apply_warping = True
        else:
            self.warping_transformer = None
            self.apply_warping = False
        self.recalibrator = None
        self.chat_engine = chat_engine
        self.verbose = verbose
        self.prompt_setting = prompt_setting
        self.shuffle_features = shuffle_features
        self.eval_mode = eval_mode
        self.bandit_strategy = bandit_strategy
        self.bandit_budget_multiplier = bandit_budget_multiplier
        self.bandit_budget_multiplier = 3
        self.bandit_top_k = bandit_top_k
        self.conformal_quantile = conformal_quantile
        self.embedding_model_name = embedding_model_name
        self.reasoning_max_tokens = reasoning_max_tokens

        self._calibrated = False
        self._calibration_q = None
        self._calibration_semantic_mean = None
        self._embedding_model = None
        self._embedding_cache = {}

        assert isinstance(self.shuffle_features, bool), 'shuffle_features must be a boolean'
        assert self.eval_mode in ["mc", "bandit_ucb1", "bandit_ucb1_tuned", "bandit_ucb1_kl"], "Unsupported eval_mode"


    async def _async_generate(self, few_shot_template, query_example, query_idx):
        '''Generate a response from the LLM async.'''
        message = []
        message.append({"role": "system","content": "You are an AI assistant that helps people find information."})
        user_message = few_shot_template.format(Q=query_example['Q'])
        message.append({"role": "user", "content": user_message})

        MAX_RETRIES = 3

        async with ClientSession(trust_env=True) as session:
            openai.aiosession.set(session)

            resp = None
            n_preds = int(self.n_gens/self.n_templates) if self.bootstrapping else int(self.n_gens)
            for retry in range(MAX_RETRIES):
                try:
                    start_time = time.time()
                    self.rate_limiter.add_request(request_text=user_message, current_time=start_time)
                    request_kwargs = dict(
                        messages=message,
                        temperature=0.7,
                        max_tokens=8,
                        top_p=0.95,
                        n=max(n_preds, 3),            # e.g. for 5 templates, get 2 generations per template
                        request_timeout=10,
                    )
                    if API_TYPE.lower() == "azure":
                        request_kwargs["engine"] = self.chat_engine
                    else:
                        request_kwargs["model"] = self.chat_engine

                    resp = await openai.ChatCompletion.acreate(**request_kwargs)
                    self.rate_limiter.add_request(request_token_count=resp['usage']['total_tokens'], current_time=time.time())
                    break
                except Exception as e:
                    print(f'[SM] RETRYING LLM REQUEST {retry+1}/{MAX_RETRIES}...')
                    print(resp)
                    if retry == MAX_RETRIES-1:
                        await openai.aiosession.get().close()
                        raise e
                    pass

        await openai.aiosession.get().close()

        if resp is None:
            return None

        tot_tokens = resp['usage']['total_tokens']
        tot_cost = 0.0015*(resp['usage']['prompt_tokens']/1000) + 0.002*(resp['usage']['completion_tokens']/1000)

        return query_idx, resp, tot_cost, tot_tokens



    async def _generate_concurrently(self, few_shot_templates, query_examples):
        '''Perform concurrent generation of responses from the LLM async.'''

        coroutines = []
        for template in few_shot_templates:
            for query_idx, query_example in enumerate(query_examples):
                coroutines.append(self._async_generate(template, query_example, query_idx))

        tasks = [asyncio.create_task(c) for c in coroutines]

        results = [[] for _ in range(len(query_examples))]      # nested list

        llm_response = await asyncio.gather(*tasks)

        for response in llm_response:
            if response is not None:
                query_idx, resp, tot_cost, tot_tokens = response
                results[query_idx].append([resp, tot_cost, tot_tokens])

        return results  # format [(resp, tot_cost, tot_tokens), None, (resp, tot_cost, tot_tokens)]

    
    async def _predict(self, all_prompt_templates, query_examples):
        start = time.time()
        all_preds = []
        tot_tokens = 0
        tot_cost = 0

        bool_pred_returned = []

        # make predictions in chunks of 5, for each chunk make concurent calls
        for i in range(0, len(query_examples), 5):
            query_chunk = query_examples[i:i+5]
            chunk_results = await self._generate_concurrently(all_prompt_templates, query_chunk)
            bool_pred_returned.extend([1 if x is not None else 0 for x in chunk_results])                # track effective number of predictions returned

            for _, sample_response in enumerate(chunk_results):
                if not sample_response:     # if sample prediction is an empty list :(
                    sample_preds = [np.nan] * self.n_gens
                else:
                    sample_preds = []
                    all_gens_text = [x['message']['content'] for template_response in sample_response for x in template_response[0]['choices'] ]        # fuarr this is some high level programming
                    for gen_text in all_gens_text:
                        gen_pred = re.findall(r"## (-?[\d.]+) ##", gen_text)
                        if len(gen_pred) == 1:
                            sample_preds.append(float(gen_pred[0]))
                        else:
                            sample_preds.append(np.nan)
                            
                    while len(sample_preds) < self.n_gens:
                        sample_preds.append(np.nan)

                    tot_cost += sum([x[1] for x in sample_response])
                    tot_tokens += sum([x[2] for x in sample_response])
                all_preds.append(sample_preds)
        
        end = time.time()
        time_taken = end - start

        success_rate = sum(bool_pred_returned)/len(bool_pred_returned)

        all_preds = np.array(all_preds).astype(float)
        y_mean = np.nanmean(all_preds, axis=1)
        y_std = np.nanstd(all_preds, axis=1)

        # Capture failed calls - impute None with average predictions
        y_mean[np.isnan(y_mean)]  = np.nanmean(y_mean)
        y_std[np.isnan(y_std)]  = np.nanmean(y_std)
        y_std[y_std<1e-5] = 1e-5  # replace small values to avoid division by zero

        return y_mean, y_std, success_rate, tot_cost, tot_tokens, time_taken

    def _ensure_embedding_model(self):
        if self._embedding_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except Exception as e:
            raise ImportError(
                "sentence-transformers is required for bandit reasoning mode. "
                "Install it with: pip install sentence-transformers"
            ) from e
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._embedding_model = SentenceTransformer(self.embedding_model_name, device=device)

    def _encode_reasonings(self, reasonings):
        self._ensure_embedding_model()
        new_texts = [r for r in reasonings if r not in self._embedding_cache]
        if new_texts:
            vectors = self._embedding_model.encode(new_texts, convert_to_numpy=True, normalize_embeddings=True)
            for text, vec in zip(new_texts, vectors):
                self._embedding_cache[text] = vec
        return [self._embedding_cache[r] for r in reasonings]

    def _pairwise_cosine_mean(self, reasonings):
        if len(reasonings) < 2:
            return 1.0
        print(f'\n\n\n[SM] Calculating pairwise cosine similarity for {len(reasonings)} reasonings')
        vectors = self._encode_reasonings(reasonings)
        print(f'\n\n\n[SM] Encoded reasonings into vectors of shape {vectors[0].shape}')
        sims = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sims.append(float(np.dot(vectors[i], vectors[j])))
        if not sims:
            return 1.0
        return float(np.mean(sims))

    async def _async_generate_with_reasoning(self, prompt_text):
        message = []
        message.append({"role": "system","content": "You are an AI assistant that helps people find information."})
        message.append({"role": "user", "content": prompt_text})

        MAX_RETRIES = 3
        async with ClientSession(trust_env=True) as session:
            openai.aiosession.set(session)

            resp = None
            for retry in range(MAX_RETRIES):
                try:
                    start_time = time.time()
                    self.rate_limiter.add_request(request_text=prompt_text, current_time=start_time)
                    request_kwargs = dict(
                        messages=message,
                        temperature=0.7,
                        max_tokens=self.reasoning_max_tokens,
                        top_p=0.95,
                        n=1,
                        request_timeout=10,
                    )
                    if API_TYPE.lower() == "azure":
                        request_kwargs["engine"] = self.chat_engine
                    else:
                        request_kwargs["model"] = self.chat_engine
                    resp = await openai.ChatCompletion.acreate(**request_kwargs)
                    self.rate_limiter.add_request(request_token_count=resp['usage']['total_tokens'], current_time=time.time())
                    break
                except Exception as e:
                    print(f'[SM] RETRYING LLM REQUEST {retry+1}/{MAX_RETRIES}...')
                    print(resp)
                    if retry == MAX_RETRIES-1:
                        await openai.aiosession.get().close()
                        raise e
                    pass

        await openai.aiosession.get().close()
        if resp is None:
            return None

        tot_tokens = resp['usage']['total_tokens']
        tot_cost = 0.0015*(resp['usage']['prompt_tokens']/1000) + 0.002*(resp['usage']['completion_tokens']/1000)
        return resp, tot_cost, tot_tokens

    def _parse_reasoning_and_score(self, text):
        score = np.nan
        reasoning = ""

        score_idx = text.find("Score:")
        reasoning_idx = text.find("Reasoning:")

        if score_idx != -1:
            score_part = text[score_idx + len("Score:"):].strip()
            if reasoning_idx != -1 and reasoning_idx > score_idx:
                score_part = text[score_idx + len("Score:"):reasoning_idx].strip()

            if "##" in score_part:
                parts = score_part.split("##")
                if len(parts) >= 2:
                    num_str = parts[1].strip()
                    try:
                        score = float(num_str)
                    except Exception:
                        score = np.nan
            else:
                num_str = score_part.split()[0] if score_part else ""
                try:
                    score = float(num_str)
                except Exception:
                    score = np.nan

        if reasoning_idx != -1:
            reasoning_part = text[reasoning_idx + len("Reasoning:"):].strip()
            if score_idx != -1 and score_idx > reasoning_idx:
                reasoning_part = text[reasoning_idx + len("Reasoning:"):score_idx].strip()
            reasoning = reasoning_part
        elif score_idx != -1:
            reasoning = text[:score_idx].strip()

        return reasoning, score

    def _initialize_conformal_calibration(self, observed_configs, observed_fvals):
        if self._calibrated:
            return 0.0, 0.0

        print("\n\n\nCalibration started\n\n\n")

        all_prompt_templates, query_examples = gen_prompt_tempates_with_reasoning(
            self.task_context,
            observed_configs,
            observed_fvals,
            observed_configs,
            n_prompts=self.n_templates,
            bootstrapping=self.bootstrapping,
            use_context=self.prompt_setting if self.prompt_setting is not None else 'full_context',
            use_feature_semantics=True,
            shuffle_features=self.shuffle_features,
            apply_warping=self.apply_warping,
        )

        print("\n\n\nGot response of calibration prompt\n\n\n")

        total_cost = 0.0
        total_time = 0.0
        reasonings = []
        residuals = []

        for i, query_example in enumerate(query_examples):
            prompt = all_prompt_templates[i % len(all_prompt_templates)].format(Q=query_example['Q'])
            start_time = time.time()
            resp_tuple = asyncio.run(self._async_generate_with_reasoning(prompt))
            if resp_tuple is None:
                continue
            resp, cost, _ = resp_tuple
            total_cost += cost
            total_time += time.time() - start_time

            content = resp['choices'][0]['message']['content']
            reasoning, pred = self._parse_reasoning_and_score(content)
            reasonings.append(reasoning)

            true_val = observed_fvals.iloc[i, 0]
            if not np.isnan(pred):
                residuals.append(float(np.abs(pred - true_val)))

        if residuals:
            self._calibration_q = float(np.quantile(residuals, self.conformal_quantile))
        else:
            self._calibration_q = 0.0
        self._calibration_semantic_mean = self._pairwise_cosine_mean(reasonings) if reasonings else 1.0
        self._calibrated = True
        print(f"\n\n\nCalibration completed. q={self._calibration_q:.6f}, Mcal={self._calibration_semantic_mean:.6f}\n\n\n")
        return total_cost, total_time

    def _ucb_value(self, mean, count, t, var_est=None):
        if count == 0:
            return float('inf')
        if self.bandit_strategy == "ucb1":
            return mean + np.sqrt(2 * np.log(t) / count)
        # ucb1_tuned
        if self.bandit_strategy == "ucb1_tuned":
            v = 0.0 if var_est is None else var_est
            v = min(0.25, v + np.sqrt(2 * np.log(t) / count))
            return mean + np.sqrt((np.log(t) / count) * v)
        if self.bandit_strategy == "ucb1_kl":
            return self._kl_ucb(mean, count, t)
        else:
            raise ValueError(f"Unsupported bandit strategy: {self.bandit_strategy}")

    def _kl_divergence(self, p, q):
        eps = 1e-12
        p = min(max(p, eps), 1 - eps)
        q = min(max(q, eps), 1 - eps)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def _kl_ucb(self, mean, count, t):
        if count == 0:
            return 1.0
        # KL-UCB for bounded rewards in [0,1]
        c = 3.0
        bound = (np.log(t) + c * np.log(max(1.0, np.log(t)))) / count
        lo = mean
        hi = 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            if self._kl_divergence(mean, mid) > bound:
                hi = mid
            else:
                lo = mid
        return lo

    def _select_query_point_bandit(self, observed_configs, observed_fvals, candidate_configs):
        # initialize calibration on warmstart configs
        cal_cost, cal_time = self._initialize_conformal_calibration(observed_configs, observed_fvals)

        n_arms = candidate_configs.shape[0]
        print(f"\n\n\nStarting UCB1 bandit selection over {n_arms} candidate points\n\n\n")
        if n_arms == 0:
            raise Exception('No candidate points to evaluate')

        budget = int(self.bandit_budget_multiplier * n_arms)
        budget = max(budget, n_arms)

        counts = np.zeros(n_arms, dtype=int)
        sums_raw = np.zeros(n_arms, dtype=float)
        sumsq_raw = np.zeros(n_arms, dtype=float)
        sums_norm = np.zeros(n_arms, dtype=float)
        reasonings = [[] for _ in range(n_arms)]

        total_cost = cal_cost
        total_time = cal_time

        def reward_from_score(score):
            return -score if self.lower_is_better else score

        observed_vals = observed_fvals.iloc[:, 0].to_numpy()
        obs_min = float(np.min(observed_vals))
        obs_max = float(np.max(observed_vals))
        obs_range = obs_max - obs_min

        def normalize_score(score):
            if obs_range <= 0:
                return 0.5
            if self.lower_is_better:
                norm = 1.0 - ((score - obs_min) / obs_range)
            else:
                norm = (score - obs_min) / obs_range
            if norm < 0.0:
                return 0.0
            if norm > 1.0:
                return 1.0
            return norm

        # initial pull each arm once
        pulls = 0
        for i in range(n_arms):
            prompt = build_bandit_prompt(
                self.task_context,
                observed_configs,
                observed_fvals,
                candidate_configs.iloc[[i]],
                use_context=self.prompt_setting if self.prompt_setting is not None else 'full_context',
                use_feature_semantics=True,
                shuffle_features=self.shuffle_features,
                apply_warping=self.apply_warping,
            )
            start_time = time.time()
            resp_tuple = asyncio.run(self._async_generate_with_reasoning(prompt))
            total_time += time.time() - start_time
            if resp_tuple is None:
                continue
            resp, cost, _ = resp_tuple
            total_cost += cost
            content = resp['choices'][0]['message']['content']
            #print(f"\n\n\nUCB1 pull response (arm {i}):\n{content}\n\n\n")
            reasoning, score = self._parse_reasoning_and_score(content)
            if np.isnan(score):
                print(f"\n\n\nUCB1 pull end: arm {i} (invalid score)\nResponse: {content}\n\n")
                continue
            counts[i] += 1
            raw_reward = reward_from_score(score)
            sums_raw[i] += raw_reward
            sumsq_raw[i] += raw_reward ** 2
            sums_norm[i] += normalize_score(score)
            reasonings[i].append(reasoning)
            pulls += 1

            print(f"\n\n\nUCB1 pulled arm {i} with score {score}: pulls {pulls}/{budget}\n\n\n")
            if pulls >= budget:
                break

        t = max(1, pulls)
        while pulls < budget:
            means_raw = np.array([s / c if c > 0 else 0.0 for s, c in zip(sums_raw, counts)])
            variances_raw = np.array([
                (sumsq_raw[i] / counts[i]) - (means_raw[i] ** 2) if counts[i] > 0 else 0.0
                for i in range(n_arms)
            ])
            if self.bandit_strategy == "ucb1_kl":
                means_norm = np.array([s / c if c > 0 else 0.0 for s, c in zip(sums_norm, counts)])
                ucb_values = np.array([self._ucb_value(means_norm[i], counts[i], t, None) for i in range(n_arms)])
            else:
                ucb_values = np.array([self._ucb_value(means_raw[i], counts[i], t, variances_raw[i]) for i in range(n_arms)])
            arm = int(np.argmax(ucb_values))

            prompt = build_bandit_prompt(
                self.task_context,
                observed_configs,
                observed_fvals,
                candidate_configs.iloc[[arm]],
                use_context=self.prompt_setting if self.prompt_setting is not None else 'full_context',
                use_feature_semantics=True,
                shuffle_features=self.shuffle_features,
                apply_warping=self.apply_warping,
            )
            start_time = time.time()
            resp_tuple = asyncio.run(self._async_generate_with_reasoning(prompt))
            total_time += time.time() - start_time
            if resp_tuple is None:
                print(f"\n\n\nUCB1 pull end: arm {arm} (no response)\n\n\n")
                t += 1
                continue
            resp, cost, _ = resp_tuple
            total_cost += cost
            content = resp['choices'][0]['message']['content']
            #print(f"\n\n\nUCB1 pull response (arm {arm}):\n{content}\n\n\n")
            reasoning, score = self._parse_reasoning_and_score(content)
            if np.isnan(score):
                print(f"\n\n\nUCB1 pull end: arm {arm} (invalid score)\nResponse: {content}\n\n\n")
                t += 1
                continue
            counts[arm] += 1
            raw_reward = reward_from_score(score)
            sums_raw[arm] += raw_reward
            sumsq_raw[arm] += raw_reward ** 2
            sums_norm[arm] += normalize_score(score)
            reasonings[arm].append(reasoning)
            pulls += 1
            t += 1

            print(f"\n\n\nUCB1 pulled arm {arm} with score {score}: pulls {pulls}/{budget}\n\n\n")
        
        # compute mean predicted score per arm in original scale
        if self.lower_is_better:
            mean_scores = np.array([-(s / c) if c > 0 else np.inf for s, c in zip(sums_raw, counts)])
        else:
            mean_scores = np.array([s / c if c > 0 else -np.inf for s, c in zip(sums_raw, counts)])

        top_k = self.bandit_top_k if self.bandit_top_k is not None else min(3, n_arms)
        top_k = max(1, min(top_k, n_arms))
        if self.lower_is_better:
            top_indices = np.argsort(mean_scores)[:top_k]
        else:
            top_indices = np.argsort(-mean_scores)[:top_k]

        print("\n\n\nTop-k arms:\n")
        for idx in top_indices:
            print(f"Arm ID: {idx}, Pulls: {counts[idx]}, Mean Value: {mean_scores[idx]}")

        print("\n\n\nUncertainty computation started\n\n\n")
        # semantic conformal selection
        best_idx = None
        best_value = None
        for idx in top_indices:
            s_new = self._pairwise_cosine_mean(reasonings[idx]) if reasonings[idx] else self._calibration_semantic_mean
            if s_new <= 0:
                s_new = self._calibration_semantic_mean if self._calibration_semantic_mean else 1.0
            lam = (self._calibration_semantic_mean / s_new) if s_new else 1.0
            adjusted = mean_scores[idx] + (self._calibration_q * lam)
            if self.lower_is_better:
                adjusted = -mean_scores[idx] + (self._calibration_q * lam)
            if best_value is None or adjusted > best_value:
                best_value = adjusted
                best_idx = idx
            
            print(f"Arm ID: {idx}, Mean Score: {mean_scores[idx]}, Mcal: {s_new:.6f}, Lambda: {lam:.6f}, Adjusted Value: {adjusted:.6f}")

        print(f"\n\n\nUncertainty computation completed, best arm is {best_idx} with adjusted value {best_value:.6f}\n\n\n")

        if best_idx is None:
            best_idx = int(np.argmax(mean_scores)) if not self.lower_is_better else int(np.argmin(mean_scores))

        best_point = candidate_configs.iloc[[best_idx], :]
        return best_point, total_cost, total_time
    
    async def _evaluate_candidate_points(self, observed_configs, observed_fvals, candidate_configs, 
                                         use_context='full_context', use_feature_semantics=True, return_ei=False):
        '''Evaluate candidate points using the LLM model.'''

        if self.prompt_setting is not None:
            use_context = self.prompt_setting

        all_run_cost = 0
        all_run_time = 0

        tot_cost = 0
        time_taken = 0

        if self.use_recalibration and self.recalibrator is None:
            recalibrator, tot_cost, time_taken = await self._get_recalibrator(observed_configs, observed_fvals)
            if recalibrator is not None:
                self.recalibrator = recalibrator
            else:
                self.recalibrator = None
            print('[Recalibration] COMPLETED')

        all_run_cost += tot_cost
        all_run_time += time_taken

        all_prompt_templates, query_examples = gen_prompt_tempates(self.task_context, observed_configs, observed_fvals, candidate_configs, 
                                                                    n_prompts=self.n_templates, bootstrapping=self.bootstrapping,
                                                                    use_context=use_context, use_feature_semantics=use_feature_semantics, 
                                                                    shuffle_features=self.shuffle_features, apply_warping=self.apply_warping)

        print('*'*100)
        print(f'Number of all_prompt_templates: {len(all_prompt_templates)}')
        print(f'Number of query_examples: {len(query_examples)}')
        print(all_prompt_templates[0].format(Q=query_examples[0]['Q']))

        response = await self._predict(all_prompt_templates, query_examples)

        y_mean, y_std, success_rate, tot_cost, tot_tokens, time_taken = response

        if self.recalibrator is not None:
            recalibrated_res = self.recalibrator(y_mean, y_std, 0.68)   # 0.68 coverage for 1 std
            y_std = np.abs(recalibrated_res.upper - recalibrated_res.lower)/2

        all_run_cost += tot_cost
        all_run_time += time_taken

        if not return_ei:
            return y_mean, y_std, all_run_cost, all_run_time
    
        else:
            # calcualte ei
            if self.lower_is_better:
                best_fval = np.min(observed_fvals.to_numpy())
                delta = -1*(y_mean - best_fval)
            else:
                best_fval = np.max(observed_fvals.to_numpy())
                delta = y_mean - best_fval
            with np.errstate(divide='ignore'):  # handle y_std=0 without warning
                Z = delta/y_std
            ei = np.where(y_std>0, delta * norm.cdf(Z) + y_std * norm.pdf(Z), 0)

            return ei, y_mean, y_std, all_run_cost, all_run_time

    def select_query_point(self, observed_configs, observed_fvals, candidate_configs):
        '''Select the next query point using expected improvement.'''

        # warp
        if self.warping_transformer is not None:
            observed_configs = self.warping_transformer.warp(observed_configs)
            candidate_configs = self.warping_transformer.warp(candidate_configs)

        if self.eval_mode == "mc":
            y_mean, y_std, cost, time_taken = asyncio.run(
                self._evaluate_candidate_points(observed_configs, observed_fvals, candidate_configs)
            )
            if self.lower_is_better:
                best_fval = np.min(observed_fvals.to_numpy())
                delta = -1*(y_mean - best_fval)
            else:
                best_fval = np.max(observed_fvals.to_numpy())
                delta = y_mean - best_fval

            with np.errstate(divide='ignore'):  # handle y_std=0 without warning
                Z = delta/y_std

            ei = np.where(y_std>0, delta * norm.cdf(Z) + y_std * norm.pdf(Z), 0)
            best_point_index = np.argmax(ei)
            best_point = candidate_configs.iloc[[best_point_index], :]
        else:
            best_point, cost, time_taken = self._select_query_point_bandit(
                observed_configs, observed_fvals, candidate_configs
            )

        # unwarp
        if self.warping_transformer is not None:
            candidate_configs = self.warping_transformer.unwarp(candidate_configs)

        if self.warping_transformer is not None:
            best_point = self.warping_transformer.unwarp(best_point)

        return best_point, cost, time_taken
