from transformers import TrainerCallback
import os

class EvalEveryNStepsCallback(TrainerCallback):
    def __init__(self, eval_steps=1000, metric="eval_loss", greater_is_better=False, save_dir_name="checkpoint-best"):
        self.eval_steps = int(eval_steps)
        self.metric = metric
        self.greater_is_better = bool(greater_is_better)
        self.best_val = None
        self.save_dir_name = save_dir_name

    def _is_better(self, new, best):
        if best is None:
            return True
        return (new > best) if self.greater_is_better else (new < best)

    # 每 N step 要求做一次 evaluate（不依賴 evaluation_strategy）
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step and state.global_step % self.eval_steps == 0:
            control.should_evaluate = True   # 交給 Trainer 的訓練迴圈去跑 evaluate()

    # evaluate 完成後會呼叫這裡；用 metrics 判斷是否刷新 SOTA，並儲存最佳模型
    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics", {})  # {'eval_loss': ..., ...}
        val = metrics.get(self.metric, None)
        if val is None:
            print(f"[EvalCB] Metric '{self.metric}' not found. Available: {list(metrics.keys())}")
            return

        if self._is_better(val, self.best_val):
            self.best_val = val
            best_dir = os.path.join(args.output_dir, self.save_dir_name)
            os.makedirs(best_dir, exist_ok=True)

            model = kwargs.get("model", None)
            if model is not None:
                model.save_pretrained(best_dir)
            tok = kwargs.get("tokenizer", None)
            if tok is not None:
                tok.save_pretrained(best_dir)

            # 標記目前這個 checkpoint 也該被保存（可有可無，save_strategy 仍照常運作）
            control.should_save = True
            print(f"[EvalCB] New best {self.metric}={val:.6f} @ step {state.global_step} -> saved to {best_dir}")
