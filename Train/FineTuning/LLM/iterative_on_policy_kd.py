# """
# 文件: iterative_on_policy_kd.py
# 功能: 支持 5 种通信业务场景的迭代式策略蒸馏 (Multi-Scenario On-Policy KD)
#
# 覆盖场景：
# 1. 网络监控 (NET_MONITOR): 指标采集 -> 异常检测 -> 阈值告警
# 2. 告警运维 (ALARM_OPS): 告警过滤 -> 关联收敛 -> 派发预案
# 3. 故障诊断 (FAULT_DIAG): 拓扑溯源 -> 信令抓包 -> 根因定界
# 4. 工单运维 (TICKET_OPS): 工单解析 -> SLA评估 -> 自动化执行 -> 归档闭环
# 5. 网络优化 (NET_OPTIM): KPI恶化分析 -> 参数仿真调优 -> 射频天馈调整 -> 效果复测
#
# 核心机制：
# - 场景路由状态机 (Scenario-Aware State Machine) 规则校验
# - 多场景并发 On-Policy 自回归采样 (Rollout)
# - 动态黄金标准链与偏航惩罚对构建 (DPO Preference Pairs)
# - 迭代式策略优化更新 (Iterative Policy Distillation)
# """
#
# import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # ============================================================
# # 1. 通信 5 大场景状态机与 Token 词表定义
# # ============================================================
#
# TOKEN_NAMES = {
#     # 场景触发 Prompt
#     1: "[P_NET_MONITOR]",
#     2: "[P_ALARM_OPS]",
#     3: "[P_FAULT_DIAG]",
#     4: "[P_TICKET_OPS]",
#     5: "[P_NET_OPTIM]",
#
#     # 1. 网络监控节点
#     11: "<metric_collect>",     # 指标流式采集
#     12: "<anomaly_detect>",      # 动态基线异常检测
#     13: "<threshold_alert>",     # 越限告警触发
#
#     # 2. 告警运维节点
#     21: "<alarm_filter>",        # 告警降噪过滤
#     22: "<alarm_correlate>",     # 多源告警根因收敛
#     23: "<dispatch_action>",     # 自动派发处理
#
#     # 3. 故障诊断节点
#     31: "<topo_trace>",          # 5GC/传输网拓扑溯源
#     32: "<sig_packet_inspect>",  # 核心网信令抓包分析
#     33: "<root_cause_locate>",   # 根因故障定界
#
#     # 4. 工单运维节点
#     41: "<ticket_parse>",        # 自然语言工单解析
#     42: "<sla_evaluate>",        # 故障等级与SLA判定
#     43: "<auto_execute>",        # 自动化预案执行
#     44: "<ticket_close>",        # 结果校验与工单归档
#
#     # 5. 网络优化节点
#     51: "<kpi_analysis>",        # 覆盖/质差KPI劣化分析
#     52: "<param_simulate>",      # 功率/切换门限仿真
#     53: "<rf_optimize>",         # 射频参数与天馈调整
#     54: "<effect_verify>",       # 优化效果增益复测
#
#     # 终止符
#     99: "<end>"
# }
#
# # 场景配置：定义每个场景的 Prompt、黄金标准推理链以及严格的状态转移图
# SCENARIO_CONFIGS = {
#     "NET_MONITOR": {
#         "name": "网络监控",
#         "prompt_token": 1,
#         "gold_chain": [1, 11, 12, 13, 99],
#         "valid_transitions": {1: [11], 11: [12], 12: [13], 13: [99]}
#     },
#     "ALARM_OPS": {
#         "name": "告警运维",
#         "prompt_token": 2,
#         "gold_chain": [2, 21, 22, 23, 99],
#         "valid_transitions": {2: [21], 21: [22], 22: [23], 23: [99]}
#     },
#     "FAULT_DIAG": {
#         "name": "故障诊断",
#         "prompt_token": 3,
#         "gold_chain": [3, 31, 32, 33, 99],
#         "valid_transitions": {3: [31], 31: [32], 32: [33], 33: [99]}
#     },
#     "TICKET_OPS": {
#         "name": "工单运维",
#         "prompt_token": 4,
#         "gold_chain": [4, 41, 42, 43, 44, 99],
#         "valid_transitions": {4: [41], 41: [42], 42: [43], 43: [44], 44: [99]}
#     },
#     "NET_OPTIM": {
#         "name": "网络优化",
#         "prompt_token": 5,
#         "gold_chain": [5, 51, 52, 53, 54, 99],
#         "valid_transitions": {5: [51], 51: [52], 52: [53], 53: [54], 54: [99]}
#     }
# }
#
#
# def multi_scenario_state_validator(token_seq, scenario_key):
#     """
#     多场景状态机规则校验器：
#     依据指定场景的图流转逻辑，严格判定生成轨迹是否合法。
#     """
#     cfg = SCENARIO_CONFIGS[scenario_key]
#     transitions = cfg["valid_transitions"]
#     prompt_token = cfg["prompt_token"]
#
#     # 提取有意义的状态节点
#     meaningful_tokens = [t for t in token_seq if t in TOKEN_NAMES]
#
#     if not meaningful_tokens or meaningful_tokens[0] != prompt_token:
#         return False, f"起始节点错误 (期望场景触发符: {TOKEN_NAMES[prompt_token]})"
#
#     current_state = meaningful_tokens[0]
#     for next_state in meaningful_tokens[1:]:
#         allowed_next_states = transitions.get(current_state, [])
#         if next_state not in allowed_next_states:
#             curr_name = TOKEN_NAMES.get(current_state, str(current_state))
#             next_name = TOKEN_NAMES.get(next_state, str(next_state))
#             return False, f"非法流转: [{curr_name}] -> [{next_name}] 不符合业务规范"
#         current_state = next_state
#         if current_state == 99:
#             break
#
#     if current_state != 99:
#         return False, "轨迹未闭环 (缺少 <end> 终止节点)"
#
#     return True, "轨迹完全合法合规"
#
#
# # ============================================================
# # 2. 策略模型结构定义 (PolicyLM)
# # ============================================================
#
# class PolicyLM(nn.Module):
#     def __init__(self, vocab_size=120, hidden_dim=128):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, hidden_dim)
#         self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
#         self.lm_head = nn.Linear(hidden_dim, vocab_size)
#
#     def forward(self, input_ids):
#         x = self.embedding(input_ids)
#         gru_out, _ = self.gru(x)
#
#         # 映射到词表分布 [Batch, Seq_Len, vocab_size]
#         logits = self.lm_head(gru_out)
#         return logits
#
#     def generate(self, prompt_ids, max_new_tokens=8, temperature=0.8):
#         """
#         自回归 Rollout 采样
#         """
#         self.eval()
#         curr_ids = prompt_ids.clone()
#         for _ in range(max_new_tokens):
#             with torch.no_grad():
#                 logits = self.forward(curr_ids)[:, -1, :] / max(temperature, 1e-5)
#                 probs = F.softmax(logits, dim=-1)
#                 next_token = torch.multinomial(probs, num_samples=1)
#                 curr_ids = torch.cat([curr_ids, next_token], dim=1)
#                 if next_token.item() == 99:
#                     break
#         return curr_ids
#
#
# # ============================================================
# # 3. 策略损失计算 (DPO 偏好对齐损失)
# # ============================================================
#
# def compute_sequence_log_probs(model, seq_ids):
#     """
#     计算输入序列各 Token 条件对数概率之和
#     """
#     logits = model(seq_ids[:, :-1])
#     labels = seq_ids[:, 1:]
#     log_probs = F.log_softmax(logits, dim=-1)
#     selected_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
#     return selected_log_probs.sum(dim=-1)
#
#
# def compute_dpo_loss(student_model, ref_model, chosen_seq, rejected_seq, beta=0.1):
#     """
#     多场景 DPO 偏好损失计算
#     """
#     pi_chosen_logp = compute_sequence_log_probs(student_model, chosen_seq)
#     pi_rejected_logp = compute_sequence_log_probs(student_model, rejected_seq)
#
#     with torch.no_grad():
#         ref_chosen_logp = compute_sequence_log_probs(ref_model, chosen_seq)
#         ref_rejected_logp = compute_sequence_log_probs(ref_model, rejected_seq)
#
#     chosen_ratio = pi_chosen_logp - ref_chosen_logp
#     rejected_ratio = pi_rejected_logp - ref_rejected_logp
#
#     loss = -F.logsigmoid(beta * (chosen_ratio - rejected_ratio)).mean()
#     return loss
#
#
# # ============================================================
# # 4. 多场景迭代式策略蒸馏训练流水线
# # ============================================================
#
# def run_multi_scenario_distillation(num_iterations=3, samples_per_scenario=6):
#     torch.manual_seed(42)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     VOCAB_SIZE = 120
#
#     student_model = PolicyLM(vocab_size=VOCAB_SIZE, hidden_dim=128).to(device)
#     optimizer = torch.optim.AdamW(student_model.parameters(), lr=3e-3)
#
#     print("=================================================================")
#     print("      启动通信 5 大场景迭代式策略蒸馏 (Multi-Scenario On-Policy KD)     ")
#     print("=================================================================")
#
#     for iter_idx in range(num_iterations):
#         print(f"\n>>>>>>>> [Iteration {iter_idx + 1}/{num_iterations}] <<<<<<<<")
#
#         # 1. 冻结当前策略作为 Reference Model
#         ref_model = copy.deepcopy(student_model).to(device)
#         ref_model.eval()
#
#         all_preference_pairs = []
#         scenario_stats = {}
#
#         # 2. 遍历 5 种场景进行 On-Policy 探索与校验
#         for sc_key, sc_cfg in SCENARIO_CONFIGS.items():
#             prompt_tensor = torch.tensor([[sc_cfg["prompt_token"]]], device=device)
#             gold_chain_tensor = torch.tensor([sc_cfg["gold_chain"]], device=device)
#
#             valid_count = 0
#             # 采样多个 Rollout
#             for _ in range(samples_per_scenario):
#                 rollout_seq = student_model.generate(prompt_tensor, max_new_tokens=7, temperature=0.9)
#                 tokens = rollout_seq[0].tolist()
#
#                 is_valid, reason = multi_scenario_state_validator(tokens, sc_key)
#                 if is_valid:
#                     valid_count += 1
#                 else:
#                     # 偏航时，动态将该场景的标准黄金链配对作为 Chosen，违规 Rollout 作为 Rejected
#                     all_preference_pairs.append((gold_chain_tensor, rollout_seq))
#
#             pass_rate = (valid_count / samples_per_scenario) * 100
#             scenario_stats[sc_cfg['name']] = pass_rate
#
#         # 打印当前轮次全场景指标
#         print("[各场景合规通过率]:")
#         for sc_name, rate in scenario_stats.items():
#             print(f"  - {sc_name: <8}: {rate:5.1f}%")
#
#         if not all_preference_pairs:
#             print("\n[Early Stop] 所有场景生成均达到 100% 规则合规，提前完成收敛！")
#             break
#
#         # 3. 策略对齐参数更新 (DPO 优化)
#         student_model.train()
#         total_loss = 0.0
#         for chosen, rejected in all_preference_pairs:
#             optimizer.zero_grad()
#             loss = compute_dpo_loss(student_model, ref_model, chosen, rejected, beta=0.25)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#         avg_loss = total_loss / len(all_preference_pairs)
#         print(f"[Policy Update] 本轮修正偏航样本数: {len(all_preference_pairs)}, 平均 DPO Loss: {avg_loss:.4f}")
#
#     # ============================================================
#     # 5. 全场景最终生成质量与合规性复测
#     # ============================================================
#     print("\n=================================================================")
#     print("                 全场景蒸馏训练效果最终验证                      ")
#     print("=================================================================")
#     student_model.eval()
#
#     for sc_key, sc_cfg in SCENARIO_CONFIGS.items():
#         prompt = torch.tensor([[sc_cfg["prompt_token"]]], device=device)
#         final_seq = student_model.generate(prompt, max_new_tokens=7, temperature=0.01)  # 贪心推理
#         tokens = final_seq[0].tolist()
#         is_valid, reason = multi_scenario_state_validator(tokens, sc_key)
#
#         path_repr = " -> ".join([TOKEN_NAMES.get(t, f"<{t}>") for t in tokens])
#         status = "PASSED" if is_valid else "FAILED"
#
#         print(f"\n场景: 【{sc_cfg['name']}】 [{status}]")
#         print(f"  - 生成推理链: {path_repr}")
#         print(f"  - 校验结论  : {reason}")
#
#
# if __name__ == "__main__":
#     run_multi_scenario_distillation(num_iterations=20, samples_per_scenario=8)


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import copy


# ==========================================
# 1. 规则验证器：通信排障拓扑状态机 (Rule Validator)
# ==========================================
class TelecomTopologyValidator:
    """
    通信拓扑与信令时序状态机（硬约束）：
    0: 告警接入与协议解析
    1: 拓扑节点下钻 (AMF -> SMF -> UPF)
    2: 信令状态机溯源 (N1/N2 -> N4 鉴权/会话)
    3: 根因定位 (Root Cause Analysis)
    4: 修复动作与自愈建议
    99: 结束标记 (End of Reasoning)
    """

    def __init__(self):
        self.valid_transitions = {
            0: [1, 2],  # 接入告警后可下钻拓扑或分析信令
            1: [2, 3],  # 拓扑确认后必须下钻信令或给出根因
            2: [3],  # 信令分析后必须收敛到根因
            3: [4, 99],  # 定位根因后给出动作或结束
            4: [99]  # 动作给出后结束
        }

    def validate_trajectory(self, token_seq: List[int]) -> Tuple[bool, float, str]:
        """
        验证学生模型自回归生成的推理状态跳转
        返回: (是否通过, 规则惩罚因子/硬得分, 违规原因)
        """
        states = [t for t in token_seq if t in self.valid_transitions or t == 99]
        if not states:
            return False, 0.0, "Empty or non-state sequence"

        # 检查是否陷入死循环自省（反思死循环检测）
        if len(states) >= 3 and len(set(states[-3:])) == 1:
            return False, 0.1, "Infinite reflection loop detected"

        # 检查状态机连通性
        for i in range(len(states) - 1):
            curr_s, next_s = states[i], states[i + 1]
            if curr_s == 99:
                break
            if next_s not in self.valid_transitions.get(curr_s, []):
                return False, 0.2, f"Illegal topology jump from state {curr_s} to {next_s}"

        # 必须完成根因收敛（到达状态 3 或 4 或 99）
        if not any(s in [3, 4, 99] for s in states):
            return False, 0.3, "Premature termination without Root Cause"

        return True, 1.0, "Valid reasoning chain"


# ==========================================
# 2. 教师模型判别器 (Teacher as a Judge)
# ==========================================
class TeacherJudge:
    """
    Qwen3-235B-Thinking 作为 Judge / Verifier
    评估推理链的逻辑深度、协议理解准确性与幻觉程度
    """

    def __init__(self, teacher_model=None):
        self.teacher = teacher_model

    def evaluate_batch(self, prompt: str, candidates: List[str]) -> List[float]:
        """
        对学生生成的 N 条候选推理轨迹进行语义打分 [0.0 ~ 1.0]
        在实际分布式工程中，此处可通过 RPC/MindIE 批处理调用 235B 教师模型
        """
        scores = []
        for cand in candidates:
            # 模拟教师模型对通信排障逻辑深度的打分（实际项目中替换为 Teacher Prompt 判定）
            score = 0.5
            if "5GC" in cand or "AMF" in cand:
                score += 0.2
            if "Root Cause" in cand:
                score += 0.2
            if "Hallucination" in cand or "Loop" in cand:
                score -= 0.4
            scores.append(max(0.0, min(1.0, score)))
        return scores


# ==========================================
# 3. 混合裁判系统 (Hybrid Judge)
# ==========================================
class HybridJudge:
    def __init__(self, rule_weight: float = 0.6, teacher_weight: float = 0.4):
        self.rule_validator = TelecomTopologyValidator()
        self.teacher_judge = TeacherJudge()
        self.w_rule = rule_weight
        self.w_teacher = teacher_weight

    def score_and_rank(self, prompt: str, rollouts: List[Dict]) -> Tuple[Dict, Dict, bool]:
        """
        输入: 学生的多个 Rollout (包含 token_ids 和 text)
        输出: (Chosen_Rollout, Rejected_Rollout, 是否构成有效偏好对)
        """
        hybrid_scores = []
        raw_texts = [r["text"] for r in rollouts]
        teacher_scores = self.teacher_judge.evaluate_batch(prompt, raw_texts)

        for i, r in enumerate(rollouts):
            is_valid, rule_score, reason = self.rule_validator.validate_trajectory(r["token_ids"])
            t_score = teacher_scores[i]

            # 综合 Reward：若硬规则违背严重，直接大幅惩罚
            final_reward = (self.w_rule * rule_score) + (self.w_teacher * t_score)
            if not is_valid:
                final_reward *= 0.5  # 硬规则拦截惩罚

            hybrid_scores.append((final_reward, r, is_valid))

        # 按综合得分降序排序
        hybrid_scores.sort(key=lambda x: x[0], reverse=True)

        best_score, chosen, best_valid = hybrid_scores[0]
        worst_score, rejected, _ = hybrid_scores[-1]

        # 构造有效偏好对的条件：存在显著得分差 (Margin) 且最优路径合法
        margin = best_score - worst_score
        if margin > 0.25 and best_valid:
            return chosen, rejected, True
        return chosen, rejected, False


# ==========================================
# 4. 迭代式 DPO 蒸馏训练器 (Iterative On-Policy Trainer)
# ==========================================
class TelecomIterativeDPOTrainer:
    def __init__(self, student_model: nn.Module, beta: float = 0.1, lr: float = 5e-6):
        self.student = student_model
        self.ref_model = copy.deepcopy(student_model)  # 参考策略 π_ref
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.beta = beta
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr)
        self.hybrid_judge = HybridJudge()

    def get_log_probs(self, model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算序列在指定模型下的对数概率和"""
        logits = model(input_ids)  # [B, Seq_Len, Vocab]
        log_probs = F.log_softmax(logits, dim=-1)
        # Shift tokens for autoregressive loss
        shift_logits = log_probs[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fn = nn.NLLLoss(reduction='none')
        # [B, Seq_Len-1]
        token_log_probs = -loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_log_probs = token_log_probs.view(shift_labels.size())
        return token_log_probs.sum(dim=-1)

    def compute_dpo_loss(self, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor) -> torch.Tensor:
        """
        标准 DPO 损失计算:
        L_DPO = -E [ log \sigma ( \beta * log(pi/ref)_w - \beta * log(pi/ref)_l ) ]
        """
        # 计算当前学生模型 π_θ 的对数概率
        pi_chosen_logp = self.get_log_probs(self.student, chosen_ids, chosen_ids)
        pi_rejected_logp = self.get_log_probs(self.student, rejected_ids, rejected_ids)

        # 计算参考模型 π_ref 的对数概率
        with torch.no_grad():
            ref_chosen_logp = self.get_log_probs(self.ref_model, chosen_ids, chosen_ids)
            ref_rejected_logp = self.get_log_probs(self.ref_model, rejected_ids, rejected_ids)

        # 对数比值 (Log Ratios)
        pi_logratios = pi_chosen_logp - pi_rejected_logp
        ref_logratios = ref_chosen_logp - ref_rejected_logp

        logits = self.beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()
        return loss

    def train_iteration_round(self, prompt_pool: List[str], rollouts_per_prompt: int = 4):
        """
        执行一轮完整的 On-Policy 迭代闭环：
        1. 采样 (Rollout) -> 2. 混合评判 (Hybrid Judge) -> 3. 构造偏好对 -> 4. DPO 更新
        """
        print("\n=== 开始新一轮迭代 (New Iteration Round) ===")
        preference_dataset = []

        # ----------------------------------------------------
        # Step 1 & 2: 学生自回归 Rollout + Hybrid Judge 过滤排序
        # ----------------------------------------------------
        self.student.eval()
        for prompt in prompt_pool:
            rollouts = []
            for _ in range(rollouts_per_prompt):
                # 模拟 32B 学生的自回归 Rollout 生成过程
                # 实际落地调用: self.student.generate(..., temperature=0.8, top_p=0.95)
                mock_tokens = [0, 1, 2, 3, 99] if torch.rand(1).item() > 0.3 else [0, 1, 1, 1]  # 模拟偏航
                mock_text = f"Prompt: {prompt} | Steps: {'->'.join(map(str, mock_tokens))}"
                rollouts.append({"token_ids": mock_tokens, "text": mock_text})

            # 调用形式 A + 规则验证器 打分并提取偏好对
            chosen, rejected, is_valid_pair = self.hybrid_judge.score_and_rank(prompt, rollouts)
            if is_valid_pair:
                # 转换为 Tensor 供训练
                c_tensor = torch.tensor([chosen["token_ids"]], dtype=torch.long)
                r_tensor = torch.tensor([rejected["token_ids"]], dtype=torch.long)
                preference_dataset.append((c_tensor, r_tensor))

        print(f"成功构建高质量偏好对数量: {len(preference_dataset)} / {len(prompt_pool)}")

        # ----------------------------------------------------
        # Step 3: DPO 策略纠偏与梯度更新 (On-Policy KD Update)
        # ----------------------------------------------------
        self.student.train()
        total_loss = 0.0
        for chosen_ids, rejected_ids in preference_dataset:
            self.optimizer.zero_grad()
            loss = self.compute_dpo_loss(chosen_ids, rejected_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(preference_dataset))
        print(f"本轮迭代 DPO Loss 均值: {avg_loss:.4f}")

        # ----------------------------------------------------
        # Step 4: 策略平滑同步 (Update Reference Model)
        # ----------------------------------------------------
        self.ref_model.load_state_dict(self.student.state_dict())
        print("参考模型 π_ref 快照更新完毕，消除多步累积误差。")


# ==========================================
# 5. 模拟运行环境测试
# ==========================================
if __name__ == "__main__":
    # 构建 Mock 学生模型用于流程打通
    class MockQwenStudent(nn.Module):
        def __init__(self, vocab_size=120, d_model=64):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.lm_head = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            return self.lm_head(x)


    mock_student = MockQwenStudent()
    trainer = TelecomIterativeDPOTrainer(student_model=mock_student)

    # 模拟真实通信故障排障场景 Prompt 池
    prompts = [
        "【5GC异常】AMF 上报 PDU Session Establishment Reject, Cause #27",
        "【核心网信令】SMF 到 UPF 的 N4 接口 Heartbeat 丢失与 Session 释放",
        "【跨域故障】UPF 与 gNodeB 之间的 GTP-U 隧道丢包导致无线掉话"
    ]

    # 执行 2 轮迭代式策略蒸馏
    for r in range(1, 3):
        print(f"\n>>>>>>>>>>>>>> 迭代轮次 Iteration [{r}] <<<<<<<<<<<<<<")
        trainer.train_iteration_round(prompt_pool=prompts, rollouts_per_prompt=4)