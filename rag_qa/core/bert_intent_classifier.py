"""
BERT-BASE-CHINESE 保险意图分类器

架构: bert-base-chinese → Dropout → Linear(768 → 9)
9分类: 闲聊寒暄/条款解读/保单查询/理赔咨询/产品对比/保费试算/退保咨询/投诉建议/out_of_domain

功能:
  - 单例模式: get_bert_classifier()
  - 延迟加载: 首次 predict() 时才加载模型（避免启动时 ~400MB 内存占用）
  - 温度校准: 训练后用验证集搜索最优温度 T，推理时校准置信度
  - 批量预测: predict_batch() 10x 快于串行
  - LLM 扩增数据: generate_training_data() 用 DeepSeek 生成变体
  - CLI 工具: --train / --evaluate / --predict / --generate-data

使用:
  python -m rag_qa.core.bert_intent_classifier --predict "肺炎住院能赔吗"
  python -m rag_qa.core.bert_intent_classifier --train --data rag_qa/data/seed_data.json --output models/bert_intent
"""

import json
import math
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from base.logger import logger
from config.settings import settings


INTENT_LABELS = [
    "闲聊寒暄",
    "条款解读",
    "保单查询",
    "理赔咨询",
    "产品对比",
    "保费试算",
    "退保咨询",
    "投诉建议",
    "out_of_domain",
]

LABEL2ID = {label: i for i, label in enumerate(INTENT_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
NUM_LABELS = len(INTENT_LABELS)


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        self.labels = [LABEL2ID[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class BERTIntentClassifier:
    """BERT-BASE-CHINESE 意图分类器"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.models.bert_classifier
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False
        self._temperature = 1.0
        self._lock = threading.Lock()

    # ── 懒加载 ──────────────────────────────

    def load(self, model_dir: str = None):
        """延迟加载模型，首次 predict/train 时自动触发"""
        if self._loaded:
            return

        path = model_dir or self.model_path

        try:
            # 尝试从微调模型目录加载
            if (
                path
                and os.path.isdir(path)
                and os.path.exists(os.path.join(path, "config.json"))
            ):
                logger.info(f"[BERT] 从微调模型加载: {path}")
                self.model = AutoModelForSequenceClassification.from_pretrained(path)
                self.tokenizer = AutoTokenizer.from_pretrained(path)
                # 加载温度参数
                temp_path = os.path.join(path, "temperature.json")
                if os.path.exists(temp_path):
                    with open(temp_path) as f:
                        self._temperature = json.load(f).get("temperature", 1.0)
            else:
                # 从 HuggingFace 下载预训练 bert-base-chinese
                logger.info(f"[BERT] 从 HuggingFace 加载: {path}")
                config = AutoConfig.from_pretrained(path, num_labels=NUM_LABELS)
                config.id2label = ID2LABEL
                config.label2id = LABEL2ID
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    path, config=config
                )
                self.tokenizer = AutoTokenizer.from_pretrained(path)

            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(
                f"[BERT] 模型加载完成 device={self.device} temp={self._temperature:.2f}"
            )
        except Exception as e:
            logger.error(f"[BERT] 模型加载失败: {e}")
            raise

    # ── 预测 ────────────────────────────────

    def predict(self, query: str) -> dict:
        """单条预测 → {"intent": "理赔咨询", "confidence": 0.92}"""
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self.load()

        inputs = self.tokenizer(
            query, truncation=True, padding=True, max_length=128, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits / self._temperature
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        return {
            "intent": ID2LABEL[pred_idx],
            "confidence": round(confidence, 4),
            "all_probs": {ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)},
        }

    def predict_batch(self, queries: list[str], batch_size: int = 32) -> list[dict]:
        """批量预测，比逐条调用快 10x"""
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self.load()

        results = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits / self._temperature
                probs = F.softmax(logits, dim=-1).cpu().numpy()

            for j, prob in enumerate(probs):
                pred_idx = int(np.argmax(prob))
                results.append(
                    {
                        "intent": ID2LABEL[pred_idx],
                        "confidence": round(float(prob[pred_idx]), 4),
                        "all_probs": {
                            ID2LABEL[k]: round(float(p), 4) for k, p in enumerate(prob)
                        },
                    }
                )

        return results

    # ── 训练 ────────────────────────────────

    def train(
        self,
        texts: list[str],
        labels: list[str],
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        eval_split: float = 0.1,
        save_path: str = None,
    ):
        """微调 BERT 分类器"""
        if not self._loaded:
            self.load()

        # 数据划分
        from sklearn.model_selection import train_test_split

        if eval_split > 0:
            train_texts, eval_texts, train_labels, eval_labels = train_test_split(
                texts, labels, test_size=eval_split, stratify=labels, random_state=42
            )
        else:
            train_texts, train_labels = texts, labels
            eval_texts, eval_labels = [], []

        train_dataset = IntentDataset(train_texts, train_labels, self.tokenizer)
        eval_dataset = (
            IntentDataset(eval_texts, eval_labels, self.tokenizer)
            if eval_texts
            else None
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 优化器 + 调度器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )

        self.model.train()
        self.model.to(self.device)

        logger.info(
            f"[BERT] 开始训练 epochs={epochs} samples={len(train_texts)} batch={batch_size}"
        )

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # 验证
            eval_acc = 0
            if eval_dataset:
                eval_acc = self._evaluate_dataset(eval_dataset)
                logger.info(
                    f"[BERT] Epoch {epoch + 1}/{epochs} loss={avg_loss:.4f} eval_acc={eval_acc:.3f}"
                )
            else:
                logger.info(f"[BERT] Epoch {epoch + 1}/{epochs} loss={avg_loss:.4f}")

        self.model.eval()

        # 温度校准
        if eval_dataset:
            self._temperature = self._calibrate_temperature(eval_dataset)
            logger.info(f"[BERT] 温度校准: T={self._temperature:.3f}")

        # 保存
        if save_path:
            self._save(save_path)
            logger.info(f"[BERT] 模型已保存到 {save_path}")

    def _evaluate_dataset(self, dataset: IntentDataset) -> float:
        loader = DataLoader(dataset, batch_size=32)
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels").to(self.device)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0

    def _calibrate_temperature(self, dataset: IntentDataset) -> float:
        """在验证集上搜索最优温度参数（最小化 NLL）"""
        loader = DataLoader(dataset, batch_size=32)

        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels").to(self.device)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                all_logits.append(outputs.logits.cpu())
                all_labels.append(labels)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        temperatures = np.logspace(-0.5, 0.5, 20)
        best_temp = 1.0
        best_nll = float("inf")

        for t in temperatures:
            scaled = logits / t
            probs = F.softmax(scaled, dim=1)
            nll = F.nll_loss(torch.log(probs + 1e-8), labels).item()
            if nll < best_nll:
                best_nll = nll
                best_temp = t

        return best_temp

    def _save(self, save_path: str):
        """保存微调模型、tokenizer、温度参数"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        with open(os.path.join(save_path, "temperature.json"), "w") as f:
            json.dump({"temperature": self._temperature}, f)

    @classmethod
    def from_pretrained(cls, path: str) -> "BERTIntentClassifier":
        """从本地微调模型加载（秒级）"""
        instance = cls(model_path=path)
        instance.load(model_dir=path)
        return instance

    # ── LLM 数据扩增 ────────────────────────

    @staticmethod
    def generate_training_data(seed_path: str, output_path: str, num_variants: int = 5):
        """用 LLM 基于种子数据生成变体训练数据"""
        from base.llm_client import get_llm_client

        with open(seed_path) as f:
            seed_data = json.load(f)

        llm = get_llm_client()
        expanded = list(seed_data)  # 保留原始种子

        for item in seed_data:
            prompt = f"""你是一个数据增强助手。请为以下保险客服查询生成 {num_variants} 个语义相似但表述不同的变体。

原始查询: {item["text"]}
意图标签: {item["label"]}

要求:
1. 保持相同的意图
2. 变换句式、用词、语气
3. 模拟真实用户的口语化提问
4. 只返回 JSON 数组，格式: ["变体1", "变体2", ...]
"""
            try:
                response = llm.chat_json(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=1024,
                )
                variants = (
                    response
                    if isinstance(response, list)
                    else response.get("variants", [])
                )
                for v in variants[:num_variants]:
                    expanded.append({"text": v, "label": item["label"]})
                logger.info(
                    f"[BERT] 数据扩增: '{item['text'][:30]}' → +{len(variants)} 变体"
                )
            except Exception as e:
                logger.warning(f"[BERT] LLM 扩增失败: {e}")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(expanded, f, ensure_ascii=False, indent=2)

        logger.info(
            f"[BERT] 数据扩增完成: {len(seed_data)} → {len(expanded)} 条 → {output_path}"
        )


# ── 单例工厂 ──────────────────────────────

_bert_classifier: Optional[BERTIntentClassifier] = None


def get_bert_classifier() -> Optional[BERTIntentClassifier]:
    """获取 BERT 意图分类器单例（首次调用时加载模型）"""
    global _bert_classifier
    if _bert_classifier is None:
        try:
            _bert_classifier = BERTIntentClassifier()
            _bert_classifier.load()
        except Exception as e:
            logger.warning(f"[BERT] 分类器初始化失败，BERT 层不可用: {e}")
            _bert_classifier = None
    return _bert_classifier


# ── CLI 入口 ───────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BERT 保险意图分类器")
    sub = parser.add_subparsers(dest="command")

    # predict
    p = sub.add_parser("--predict", help="单条预测")
    p.add_argument("--query", required=True, help="查询文本")
    p.add_argument("--model", default=None, help="微调模型路径")

    # train
    t = sub.add_parser("--train", help="训练模型")
    t.add_argument("--data", required=True, help="训练数据 JSON 路径")
    t.add_argument("--output", required=True, help="输出模型目录")
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch-size", type=int, default=16)

    # evaluate
    e = sub.add_parser("--evaluate", help="评估模型")
    e.add_argument("--model", required=True, help="模型路径")
    e.add_argument("--data", required=True, help="评估数据 JSON 路径")

    # generate-data
    g = sub.add_parser("--generate-data", help="LLM 扩增训练数据")
    g.add_argument("--seed", required=True, help="种子数据 JSON 路径")
    g.add_argument("--output", required=True, help="输出路径")
    g.add_argument("--variants", type=int, default=5, help="每条种子生成的变体数")

    args = parser.parse_args()

    if args.command == "--predict":
        cls = _bert_classifier
        if not cls:
            cls = BERTIntentClassifier()
            cls.load()
        result = cls.predict(args.query)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "--train":
        with open(args.data) as f:
            data = json.load(f)
        texts = [d["text"] for d in data]
        labels = [d["label"] for d in data]
        cls = BERTIntentClassifier()
        cls.load()
        cls.train(
            texts,
            labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=args.output,
        )

    elif args.command == "--evaluate":
        cls = BERTIntentClassifier.from_pretrained(args.model)
        with open(args.data) as f:
            data = json.load(f)
        texts = [d["text"] for d in data]
        true_labels = [d["label"] for d in data]
        results = cls.predict_batch(texts)
        correct = sum(1 for r, t in zip(results, true_labels) if r["intent"] == t)
        acc = correct / len(results) if results else 0

        # 混淆矩阵
        from collections import defaultdict

        matrix = defaultdict(lambda: defaultdict(int))
        for r, t in zip(results, true_labels):
            matrix[t][r["intent"]] += 1

        print(f"Accuracy: {acc:.4f} ({correct}/{len(results)})")
        print("\nPer-class accuracy:")
        for label in INTENT_LABELS:
            total = sum(matrix[label].values())
            correct_c = matrix[label][label]
            print(
                f"  {label}: {correct_c}/{total} = {correct_c / total:.3f}"
                if total > 0
                else f"  {label}: N/A"
            )

    elif args.command == "--generate-data":
        BERTIntentClassifier.generate_training_data(
            args.seed, args.output, num_variants=args.variants
        )
