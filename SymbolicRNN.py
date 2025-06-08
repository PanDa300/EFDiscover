import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


def mean_squared_error(y_true, y_pred, multioutput='uniform_average'):
    """
    支持多维输出的MSE

    参数:
    y_true -- 真实值数组 (n_samples,) 或 (n_samples, n_outputs)
    y_pred -- 预测值数组
    multioutput -- 'uniform_average'：各维度平均
                   'raw_values'：返回各维度MSE
                    array-like：自定义各维度权重

    返回:
    mse -- 根据multioutput参数返回不同形式
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    '''if y_true.shape != y_pred.shape:
        raise ValueError(f"形状不一致: y_true {y_true.shape}, y_pred {y_pred.shape}")'''
    output_errors = np.mean(np.square(y_true - y_pred), axis=0)

    if isinstance(multioutput, (list, np.ndarray)):
        return np.dot(output_errors, multioutput)
    elif multioutput == 'raw_values':
        return output_errors
    elif multioutput == 'uniform_average':
        return np.mean(output_errors)
    else:
        raise ValueError("不支持的multioutput参数")
class SafeOperations:
    @staticmethod
    def safe_div(a, b):
        """双重保护除法"""
        denominator = np.where(np.abs(b) < 1e-6, np.sign(b)*1e-6 + 1e-12, b)
        return np.clip(a / denominator, -1e10, 1e10)  # 防止数值溢出

    @staticmethod
    def safe_log(x):
        """双重保护对数"""
        safe_x = np.clip(np.abs(x), 1e-12, 1e12)  # 限制输入范围
        return np.log(safe_x + 1e-12)

    @staticmethod
    def safe_exp(x):
        """安全指数函数"""
        return np.exp(np.clip(x, -100, 100))  # 防止exp(1000)导致溢出

class SymbolicRegressorRNN(nn.Module):
    def __init__(self, data, max_len=20, hidden_size=128):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data.values if isinstance(data, pd.DataFrame) else data
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.base_vocab = ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log', '(', ')', 'c']
        self.current_vocab = None
        self.model = None
        # 更新基础符号表
        self.safe_ops = {
            '/': SafeOperations.safe_div,
            'log': SafeOperations.safe_log,
            'exp': SafeOperations.safe_exp,
            'sqrt': lambda x: np.sqrt(np.clip(x, 0, None)),
            'abs': np.abs
        }

    def _build_vocab(self, num_features):
        """根据特征数量生成动态符号表"""
        return [f'x{i}' for i in range(num_features)] + self.base_vocab

    def _init_model(self, vocab_size):
        """初始化PyTorch模型"""
        return nn.Sequential(
            nn.Embedding(vocab_size, 64),
            nn.LSTM(64, self.hidden_size, batch_first=True),
            nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True),
            nn.Linear(self.hidden_size, vocab_size)
        )

    def _generate_safe_leaf(self, num_features):
        """生成带数值保护的叶子节点"""
        if np.random.rand() < 0.3:
            val = np.clip(np.random.uniform(-10, 10), -1e3, 1e3)  # 限制常数范围
            return f"{val:.6f}"
        else:
            return f"x{np.random.randint(num_features)}"

    def generate_expression(self, num_features, max_depth=3):
        """生成带多重保护的表达式"""

        def _generate(depth=0):
            if depth >= max_depth or np.random.rand() < 0.2:
                return self._generate_safe_leaf(num_features)

            op = np.random.choice(['+', '-', '*', '/', 'log', 'sqrt', 'exp'])

            # 增强生成保护规则
            if op == '/':
                numerator = f"({_generate(depth + 1)})"
                denominator = f"safe_div_denom({_generate(depth + 1)})"  # 特殊保护分母
                return f"safe_div({numerator}, {denominator})"
            elif op == 'log':
                arg = f"safe_log_arg({_generate(depth + 1)})"
                return f"safe_log({arg})"
            elif op == 'exp':
                arg = f"safe_exp_arg({_generate(depth + 1)})"
                return f"safe_exp({arg})"
            elif op == 'sqrt':
                arg = f"safe_sqrt_arg({_generate(depth + 1)})"
                return f"sqrt({arg})"
            else:
                return f"({_generate(depth + 1)} {op} {_generate(depth + 1)})"

        return _generate(0)

    def evaluate_expression(self, expr, X, y, verbose=False):
        """增强型安全评估"""
        try:
            # 准备带有嵌套保护的评估环境
            env = {
                'x' + str(i): np.clip(X[:, i], -1e6, 1e6) for i in range(X.shape[1])  # 限制输入范围
            }
            env.update({
                'safe_div': self.safe_ops['/'],
                'safe_div_denom': lambda x: x + 1e-6,
                'safe_log': self.safe_ops['log'],
                'safe_log_arg': lambda x: np.abs(x) + 1e-6,
                'safe_exp': self.safe_ops['exp'],
                'safe_exp_arg': lambda x: np.clip(x, -100, 100),
                'safe_sqrt_arg': lambda x: np.abs(x),
                **self.safe_ops
            })

            compiled_expr = compile(expr, '<string>', 'eval')  # 预编译提高安全性
            y_pred = eval(compiled_expr, env)

            '''# 严格有效性检查
            if not isinstance(y_pred, np.ndarray):
                raise ValueError("表达式未生成数组")'''

            valid_mask = np.isfinite(y_pred)
            if np.sum(valid_mask) < len(y_pred) * 0.95:  # 允许最多5%无效值
                raise ValueError("无效值超过阈值")

            if np.abs(y_pred).max() > 1e20:  # 防止数值过大
                raise ValueError("数值溢出")
            return mean_squared_error(y[valid_mask], y_pred[valid_mask])

        except Exception as e:
            if verbose:
                print(f"表达式 '{expr}' 失败原因: {str(e)}")
            return np.inf

    def find_best_expression(self, lhs_cols, rhs_cols, loss_threshold=0.01):
        # 提取数据
        X = self.data[:, lhs_cols]
        y = self.data[:, rhs_cols]

        # 动态初始化模型
        num_features = len(lhs_cols)
        vocab = self._build_vocab(num_features)
        if self.current_vocab != vocab:
            self.current_vocab = vocab
            self.vocab_size = len(vocab)
            self.model = self._init_model(self.vocab_size).to(self.device)

        # 搜索逻辑
        best_loss = np.inf
        for _ in range(100):
            expr = self.generate_expression(num_features)
            current_loss = self.evaluate_expression(expr, X, y)
            if current_loss < best_loss:
                best_loss = current_loss
                if best_loss <= loss_threshold:
                    break

        return {
            "feasible": best_loss <= loss_threshold,
            "best_loss": best_loss,
            "features": lhs_cols,
            "target": rhs_cols
        }


class DataProcessor:
    def __init__(self):
        # 添加数据标准化
        raw_data = np.random.rand(100, 4)
        self.df = pd.DataFrame({
            'x0': (raw_data[:,0] - 0.5) * 2,   # 范围[-1,1]
            'x1': (raw_data[:,1] - 0.5) * 2,
            'y1': np.sin(raw_data[:,2]),
            'y2': np.cos(raw_data[:,3])
        })
        self.regressor = SymbolicRegressorRNN(data=self.df)

    def analyze_combinations(self, combinations):
        results = []
        for lhs, rhs in combinations:
            result = self.regressor.find_best_expression(
                lhs_cols=[self.df.columns.get_loc(c) for c in lhs],
                rhs_cols=[self.df.columns.get_loc(c) for c in rhs]
            )
            results.append(result)
            print(f"{lhs} -> {rhs}: Loss={result['best_loss']:.4f}")
        return results


# 使用示例 --------------------------------------------------
if __name__ == "__main__":
    processor = DataProcessor()

    test_combinations = [
        (['x0', 'x1'], ['y1']),
        (['x0'], ['y2']),
    ]

    processor.analyze_combinations(test_combinations)