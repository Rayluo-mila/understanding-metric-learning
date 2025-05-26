from collections import defaultdict
import json
import os
import torch
import wandb
from termcolor import colored

FORMAT_CONFIG = {
    "rl": {
        "train": [
            ("episode", "E", "int"),
            ("step", "S", "int"),
            ("duration", "D", "time"),
            ("episode_reward", "R", "float"),
            ("batch_reward", "BR", "float"),
            ("actor_loss", "ALOSS", "float"),
            ("q_loss", "QLOSS", "float"),
            ("critic_loss", "CLOSS", "float"),
            ("metric_loss", "MLOSS", "float"),
            ("transition_loss", "TLOSS", "float"),
            ("reward_loss", "RLOSS", "float"),
            ("mu_bd", "MUBD", "float"),
            ("mu_rd", "MURD", "float"),
            ("mu_zd", "MUZD", "float"),
            ("mu_ratio", "MUR", "float"),
            ("ir", "IR", "float"),
            ("latprior_loss", "ILOSS", "float"),
            ("norm", "NORM", "float"),
            ("fps", "FPS", "float"),
        ],
        "eval": [
            ("step", "S", "int"),
            ("episode_reward", "ER", "float"),
            ("debug_duration", "D", "time"),
            ("df_squashed_DF_L2", "DFL2", "float"),
            ("df_squashed_DF", "DF", "float"),
            ("df_pos_score_L2", "POSL2", "float"),
            ("df_neg_score_L2", "NEGL2", "float"),
        ],
    }
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating, freq=1):
        self._file_name = file_name
        if os.path.exists(file_name):
            os.remove(file_name)
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._freq = freq
        self._count = 0

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def get(self, key):
        if key in self._meters:
            return self._meters[key].value()
        else:
            return None

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith("train"):
                key = key[len("train") + 1 :]
            else:
                key = key[len("eval") + 1 :]
            key = key.replace("/", "_")
            data[key] = meter.value()
        return data

    def _dump_to_file(self, data):
        with open(self._file_name, "a") as f:
            f.write(json.dumps(data) + "\n")

    def _format(self, key, value, ty):
        template = "%s: "
        if ty == "int":
            template += "%d"
        elif ty == "float":
            template += "%.04f"
        elif ty == "time":
            template += "%.01f s"
        else:
            raise "invalid format type: %s" % ty
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, "yellow" if prefix == "train" else "green")
        pieces = ["{:5}".format(prefix)]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print("| %s" % (" | ".join(pieces)))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        self._count += 1
        if self._count == self._freq:
            self._count = 0
            data = self._prime_meters()
            data["step"] = step
            self._dump_to_console(data, prefix)
            self._meters.clear()


class Logger(object):
    def __init__(
        self,
        log_dir,
        use_wandb=True,
        wandb_proj_name="Metrics",
        wandb_run_name=None,
        config="rl",
        args=None,
        wandb_freq=1,
    ):
        self._log_dir = log_dir
        self._use_wandb = use_wandb
        self._dump_freq = wandb_freq
        self._dump_count = 0
        self._use_wandb_params = set()

        if self._use_wandb:
            # wandb.require("core")
            wandb.init(
                project=wandb_proj_name,
                config=args if args else {},
                name=wandb_run_name,
                dir=log_dir,
            )

        self._train_mg = MetersGroup(
            os.path.join(log_dir, "train.log"),
            formating=FORMAT_CONFIG[config]["train"],
            freq=4,  # dump to the console every 4 dumps (average all logged metrics)
        )
        self._eval_mg = MetersGroup(
            os.path.join(log_dir, "eval.log"),
            formating=FORMAT_CONFIG[config]["eval"],
        )

    def _try_wandb_log(self, key, value, step):
        if self._use_wandb:
            wandb.log({key: value}, step=step)

    def _try_wandb_log_histogram(self, key, histogram, step):
        if self._use_wandb:
            wandb.log({key: wandb.Histogram(histogram)}, step=step)

    def _try_wandb_log_pic(self, key, pic, step):
        if self._use_wandb:
            wandb.log({key: wandb.Image(pic)}, step=step)

    def log(self, key, value, step, n=1, log_on_wandb=True):
        assert key.startswith("train") or key.startswith("eval")
        if type(value) == torch.Tensor:
            value = value.item()
        mg = self._train_mg if key.startswith("train") else self._eval_mg
        mg.log(key, value, n)
        if log_on_wandb:
            self._use_wandb_params.add(key)

    def log_param(self, key, param, step):
        self.log_histogram(key + "_w", param.weight.data, step)
        if hasattr(param.weight, "grad") and param.weight.grad is not None:
            self.log_histogram(key + "_w_g", param.weight.grad.data, step)
        if hasattr(param, "bias"):
            self.log_histogram(key + "_b", param.bias.data, step)
            if hasattr(param.bias, "grad") and param.bias.grad is not None:
                self.log_histogram(key + "_b_g", param.bias.grad.data, step)

    def log_histogram(self, key, histogram, step):
        assert key.startswith("train") or key.startswith("eval")
        self._try_wandb_log_histogram(key, histogram, step)

    def log_pic(self, key, pic, step):
        assert key.startswith("train") or key.startswith("eval")
        self._try_wandb_log_pic(key, pic, step)

    def dump(self, step):
        if self._dump_count % self._dump_freq == 0:
            for key in self._use_wandb_params:
                value = self.get(key)
                if value is not None:
                    self._try_wandb_log(key, value, step)
            self._train_mg.dump(step, "train")
            self._eval_mg.dump(step, "eval")
        self._dump_count += 1

    def get(self, key):
        mg = self._train_mg if key.startswith("train") else self._eval_mg
        value = mg.get(key)
        return value

    def update_wandb_config(self, config):
        if self._use_wandb:
            wandb.config.update(config, allow_val_change=True)

    def finish(self):
        if self._use_wandb:
            try:
                wandb.finish()
                print("wandb.finish() completed successfully.")
            except Exception as e:
                print(f"Error in wandb.finish(): {e}")
