import asyncio
import json
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict

# 实际生产中使用 OpenAI 兼容 SDK 访问本地部署的高吞吐 vLLM / MindIE 实例
from openai import AsyncOpenAI


# ==========================================
# 1. 数据结构定义
# ==========================================
@dataclass
class RawTeacherTrace:
    sample_id: str
    prompt: str
    raw_thinking: str
    raw_answer: str
    ground_truth_cause: str


@dataclass
class GoldenCoTSample:
    sample_id: str
    formatted_text: str
    raw_len: int
    purified_len: int
    compression_rate: float


# ==========================================
# 2. 自动化提纯核心流水线
# ==========================================
class IndustrialCoTPurifier:
    def __init__(self, api_base: str = "http://localhost:8000/v1", api_key: str = "EMPTY"):
        # 连接轻量重写模型 (如部署在单卡的 Qwen2.5-7B-Instruct / 14B)
        self.client = AsyncOpenAI(base_url=api_base, api_key=api_key)
        self.rewrite_model = "Qwen2.5-7B-Instruct"

        # 3GPP 核心拓扑偏序图 (DAG): 定义合法的网元排障层级顺序
        # 检查逻辑：若链条中同时存在上游和下游网元，下游网元不得大幅早于上游网元出现
        self.telecom_dag = {
            "GNB": 0,
            "AMF": 1,
            "AUSF": 1,
            "UDM": 2,
            "SMF": 2,
            "PCF": 3,
            "UPF": 4
        }

    # -----------------------------------------------------------
    # Layer 1: 毫秒级前置规则粗筛 (无需调用模型，拦截 20% 明显废样本)
    # -----------------------------------------------------------
    def _layer1_fast_filter(self, trace: RawTeacherTrace) -> bool:
        """检查基本长度、答案关键词命中与连续 Token 复读死循环"""
        # 1. 答案正确性粗筛
        if trace.ground_truth_cause not in trace.raw_answer:
            return False

        # 2. 思考链过短（未深度推理）或过长（死循环失控）
        if len(trace.raw_thinking) < 100 or len(trace.raw_thinking) > 8000:
            return False

        # 3. 粗暴死循环复读检测（30字符窗口连续重复）
        text = trace.raw_thinking
        for i in range(len(text) - 60):
            chunk = text[i:i + 30]
            if text.count(chunk) > 3:  # 同一子串出现超过3次判定为复读死循环
                return False

        return True

    # -----------------------------------------------------------
    # Layer 2: LLM 自动化蒸馏与提纯（批量重写主力，消除自省口头禅）
    # -----------------------------------------------------------
    async def _layer2_llm_rewrite(self, prompt: str, raw_think: str) -> Optional[str]:
        """利用 7B 尺寸小模型并发消除自省废话，提取黄金单向推导链"""
        system_prompt = (
            "你是一个通信核心网排障专家。请阅读用户给出的故障排查思维链，执行精简提纯：\n"
            "1. 剔除所有自我反思、纠结、推翻重来（如'等等/重新想'）等口语化试错过程；\n"
            "2. 严格保留所有网元（AMF/SMF/UPF）、协议接口（N11/N4/PFCP）、错误码及技术因果链条；\n"
            "3. 输出格式必须是一条直接、严密、单向的排障黄金推导过程，严禁包含任何前缀闲聊。"
        )
        user_content = f"【排障问题】：{prompt}\n\n【原始思考过程】：\n{raw_think}"

        try:
            response = await self.client.chat.completions.create(
                model=self.rewrite_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,  # 低温保真
                max_tokens=2048
            )
            purified_think = response.choices[0].message.content.strip()
            return purified_think
        except Exception as e:
            print(f"[API Error] 模型重写失败: {e}")
            return None

    # -----------------------------------------------------------
    # Layer 3: 粗粒度通信拓扑 DAG 校验 (业务合规兜底)
    # -----------------------------------------------------------
    def _layer3_dag_topology_check(self, purified_think: str) -> bool:
        """从重写文本中提取核心网元顺序，检验是否符合通信协议栈拓扑"""
        text_upper = purified_think.upper()

        # 记录各网元在思考链中首次出现的字符索引位置
        entity_positions = {}
        for entity in self.telecom_dag:
            idx = text_upper.find(entity)
            if idx != -1:
                entity_positions[entity] = idx

        if len(entity_positions) < 2:
            return True  # 涉及单网元，无需拓扑顺序校验

        # 按照在文本中出现的先后排序
        sorted_entities = sorted(entity_positions.keys(), key=lambda k: entity_positions[k])

        # 校验拓扑是否发生严重倒置（如 UPF 毫无理由地先于 AMF/SMF 出现）
        max_level_seen = -1
        for ent in sorted_entities:
            current_level = self.telecom_dag[ent]
            # 允许同级或向后跳转，若出现大幅向前跨级回跳且未有解释，则判为幻觉跳步
            if current_level < max_level_seen - 2:
                return False
            max_level_seen = max(max_level_seen, current_level)

        return True

    # -----------------------------------------------------------
    # 单样本全流程流水线 (Layer 1 -> Layer 2 -> Layer 3)
    # -----------------------------------------------------------
    async def process_single_trace(self, trace: RawTeacherTrace) -> Optional[GoldenCoTSample]:
        # 1. 快速粗筛
        if not self._layer1_fast_filter(trace):
            return None

        # 2. 小模型批量重写
        purified_think = await self._layer2_llm_rewrite(trace.prompt, trace.raw_thinking)
        if not purified_think:
            return None

        # 3. 拓扑 DAG 校验
        if not self._layer3_dag_topology_check(purified_think):
            return None

        # 4. 组装为标准的 ChatML 结构化蒸馏训练格式
        formatted_entry = (
            f"<|im_start|>user\n{trace.prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<think>\n{purified_think}\n</think>\n"
            f"{trace.raw_answer}<|im_end|>"
        )

        orig_len = len(trace.raw_thinking)
        puri_len = len(purified_think)
        comp_rate = round((1 - puri_len / orig_len) * 100, 2) if orig_len > 0 else 0.0

        return GoldenCoTSample(
            sample_id=trace.sample_id,
            formatted_text=formatted_entry,
            raw_len=orig_len,
            purified_len=puri_len,
            compression_rate=comp_rate
        )

    # -----------------------------------------------------------
    # 并发批处理入口 (支持超大规模语料高吞吐提纯)
    # -----------------------------------------------------------
    async def batch_pipeline(self, traces: List[RawTeacherTrace], batch_size: int = 50) -> List[GoldenCoTSample]:
        results = []
        for i in range(0, len(traces), batch_size):
            batch = traces[i:i + batch_size]
            tasks = [self.process_single_trace(t) for t in batch]
            batch_res = await asyncio.gather(*tasks)
            # 过滤掉为 None 的样本 (拒绝采样)
            valid_samples = [r for r in batch_res if r is not None]
            results.extend(valid_samples)
            print(f"进度: 已处理 {min(i + batch_size, len(traces))}/{len(traces)}，有效入库: {len(results)}")
        return results


# ==========================================
# 3. 模拟运行环境
# ==========================================
async def main():
    # 构造模拟输入数据
    mock_data = [
        RawTeacherTrace(
            sample_id="CASE_5GC_001",
            prompt="【故障工单】部分终端在发起切片业务时上报 PDU Session 建立超时，请定位根因。",
            raw_thinking="""
            收到排障需求。我们来看看告警。
            等等，让我先看无线 gNB 侧是不是有干扰？不对，先看核心网 NAS 消息。
            查看 AMF 日志，发现收到 PDU Session Establishment Request，随后 AMF 向 SMF 转发 N11 消息正常。
            慢着，刚才看错了，重新理一下...
            继续往下看，SMF 尝试通过 N4 接口向 UPF 请求建立 Session，但是 UPF 的 PFCP 响应超时。
            再想想，是不是 SMF 本身有问题？检查 SMF 负载正常。
            排查 UPF，发现 UPF 对应 N4 接口由于链路拥塞丢包率达 30%，导致 PFCP 心跳与建链请求丢失。根因清晰。
            """,
            raw_answer="【根因】：UPF 节点 N4 接口网络链路拥塞，导致 PFCP 消息交互超时。\n【处置建议】：执行用户面 N4 接口链路扩容并开启双链路主备倒换。",
            ground_truth_cause="UPF 节点 N4 接口网络链路拥塞"
        )
    ]

    # 初始化提纯器 (注：若无本地 API 服务，代码内部会捕获 API 错误演示流程)
    purifier = IndustrialCoTPurifier(api_base="http://localhost:8000/v1")

    print(">>> 启动自动化思维链提纯流水线...")
    # 由于环境无真实 vLLM 实例，此处模拟重写输出演示完整链路
    purifier._layer2_llm_rewrite = lambda p, r: asyncio.sleep(0.01, result="""1. 检查 AMF 节点 NAS 消息及 N11 接口，确认 AMF 正常向 SMF 转发 PDU Session 建立请求；
2. 深钻 SMF 与 UPF 间 N4 接口交互，发现 SMF 下发的 PFCP Session Request 未收到响应；
3. 检查 UPF 节点状态与网络接口，确认 N4 接口存在 30% 链路拥塞丢包，导致 PFCP 信令交互超时。""")

    golden_dataset = await purifier.batch_pipeline(mock_data, batch_size=10)

    if golden_dataset:
        sample = golden_dataset[0]
        print("\n" + "=" * 50)
        print(f"✅ 样本清洗完成 (ID: {sample.sample_id})")
        print(f"Token 压缩率: {sample.compression_rate}% ({sample.raw_len} -> {sample.purified_len} 字符)")
        print("\n【最终输出给学生模型训练的标准样本格式】：")
        print(sample.formatted_text)
        print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())