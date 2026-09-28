import numpy as np
from collections import namedtuple
from time import time

_VER_ = '0.9.1'


class SumTree(object):
  """
  此SumTree代码是Morvan Zhou的修改版本: 
  https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
  """
  data_pointer = 0
  stored_data = 0
  
  """
  这里我们初始化树，所有节点=0，并初始化数据，所有值=0
  """
  def __init__(self, capacity):
      self.capacity = capacity # 包含经验的叶节点（最终节点）数量
      
      # 生成所有节点值为0的树
      # 要理解这个计算(2 * capacity - 1)，请看上面的示意图
      # 记住我们在二叉节点中（每个节点最多有2个子节点），所以2倍叶子大小(capacity) - 1（根节点）
      # 父节点 = capacity - 1
      # 叶节点 = capacity
      self.tree = np.zeros(2 * capacity - 1)
      
      """ 树结构:
          0
         / \
        0   0
       / \ / \
      0  0 0  0  [大小: capacity] 这一行存储的是优先级分数(即pi)
      """
      
      # 包含经验（所以数据大小为capacity）
      self.data = np.zeros(capacity, dtype=object)
  
  
  """
  这里我们将优先级分数添加到sumtree叶子中，并将经验添加到数据中
  """
  def add(self, priority, data):
      # 查看我们要将经验放在哪个索引
      tree_index = self.data_pointer + self.capacity - 1
      
      """ 树结构:
          0
         / \
        0   0
       / \ / \
tree_index  0 0  0  我们从左到右填充叶子
      """
      
      # 更新数据帧
      self.data[self.data_pointer] = data
      
      # 更新叶子
      self.update (tree_index, priority)
      
      # data_pointer加1
      self.data_pointer += 1
      
      if self.data_pointer >= self.capacity:  # 如果超过容量，回到第一个索引（覆盖）
          self.data_pointer = 0
          
      self.stored_data += 1
          
  
  """
  更新叶子优先级分数，并将变化传播到树中
  """
  def update(self, tree_index, priority):
      # 变化 = 新优先级分数 - 旧优先级分数
      change = priority - self.tree[tree_index]
      self.tree[tree_index] = priority
      
      # 然后将变化传播到树中
      while tree_index != 0:    # 此方法比参考代码中的递归循环更快
          
          """
          这里我们要访问上一行
          这个树中的数字是索引，不是优先级值
          
              0
             / \
            1   2
           / \ / \
          3  4 5  [6] 
          
          如果我们在索引6的叶子处，我们更新了优先级分数
          然后我们需要更新索引2的节点
          所以 tree_index = (tree_index - 1) // 2
          tree_index = (6-1)//2
          tree_index = 2 (因为 // 会四舍五入)
          """
          tree_index = (tree_index - 1) // 2
          self.tree[tree_index] += change
  
  
  """
  这里我们获取leaf_index、该叶子的优先级值以及与该索引关联的经验
  """
  def get_leaf(self, v):
      """
      树结构和数组存储:
      树索引:
           0         -> 存储优先级总和
          / \
        1     2
       / \   / \
      3   4 5   6    -> 存储经验的优先级
      数组存储类型:
      [0,1,2,3,4,5,6]
      """
      parent_index = 0
      
      while True: # while循环比参考代码中的方法更快
          left_child_index = 2 * parent_index + 1
          right_child_index = left_child_index + 1
          
          # 如果到达底部，结束搜索
          if left_child_index >= len(self.tree):
              leaf_index = parent_index
              break
          
          else: # 向下搜索，总是搜索更高优先级的节点
              
              if v <= self.tree[left_child_index]:
                  parent_index = left_child_index
                  
              else:
                  v -= self.tree[left_child_index]
                  parent_index = right_child_index
          
      data_index = leaf_index - self.capacity + 1

      return leaf_index, self.tree[leaf_index], self.data[data_index]

  def get_leafs(self, values):
    indices = []
    priorities = []
    datas = []    
    for v in values:
      idx, prior, data = self.get_leaf(v)
      if 'int' in str(type(data)):
        print("\n警告: 无效叶节点 data:{}  v:{}  idx:{}  priority:{}".format(
            data, v, idx, prior))
        continue
      indices.append(idx)
      priorities.append(prior)
      datas.append(data)
    return np.array(indices), np.array(priorities), datas
    
  
  @property
  def total_priority(self):
      return self.tree[0] # 返回根节点
    
    
class GenericReplayBuffer(object):
  def __init__(self,  capacity, engine='torch', device=None, continuous=False):
    self.capacity = capacity
    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    self.engine = engine
    self.continuous = continuous
    if engine == 'torch' and device is None:
      raise ValueError("使用torch引擎时必须提供device参数")
    self.device = device
    self.episode = -1
    self.debug_cpu_copy = []
    self.cpu_start = 0
    self.cpu_end = 0
    self.__version__ = _VER_
    print("Init GRB v.{}".format(self.__version__))
    return
  

  def start_cpu_copy(self):
    self.cpu_start = time()
  
  def end_cpu_copy(self):
    self.cpu_end = time()
    self.debug_cpu_copy.append(self.cpu_end - self.cpu_start)
    return
  
  def get_cpu_copy_time(self):
    return np.sum(self.debug_cpu_copy)  
  
  def _prepare_experience_buffer(self, experience_buffer):
    np_states = np.vstack([e.state for e in experience_buffer if e is not None])
    np_actions = np.vstack([e.action for e in experience_buffer if e is not None])
    np_rewards = np.vstack([e.reward for e in experience_buffer if e is not None])
    np_next_states = np.vstack([e.next_state for e in experience_buffer if e is not None])
    np_dones = np.vstack([e.done for e in experience_buffer if e is not None]).astype(np.uint8)
    if self.engine == 'torch':
      self.start_cpu_copy()
      import torch as th
      states = th.from_numpy(np_states).float().to(self.device)
      if self.continuous:
        actions = th.from_numpy(np_actions).float().to(self.device)
      else:
        actions = th.from_numpy(np_actions).long().to(self.device)
      rewards = th.from_numpy(np_rewards).float().to(self.device)
      next_states = th.from_numpy(np_next_states).float().to(self.device)
      dones = th.from_numpy(np_dones).float().to(self.device)
      self.end_cpu_copy()
    else:
      states = np_states
      actions = np_actions
      rewards = np_rewards
      next_states = np_next_states
      dones = np_dones
    return (states, actions, rewards, next_states, dones)
    
  def add(self, state, action, reward, next_state, done):
    """添加新经验到内存。"""
    e = self.experience(state, action, reward, next_state, done)
    self.store(e)
    return

  def store(self, experience):
    raise ValueError("调用了抽象方法!")
    return
    
    
class PERMemory(GenericReplayBuffer):  # 以 ( s, a, r, s_ ) 形式存储在SumTree中
  """
  此SumTree代码是修改版本，原始代码来自:
  https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
  """

  def __init__(self, **kwargs):
    # 创建树 
    """
    记住我们的树由一个sumtree组成，其叶子包含优先级分数
    还有一个数据数组
    我们不使用deque，因为这意味着在每个时间步我们的经验索引都会改变一个。
    我们更倾向于使用简单的数组，当内存满时覆盖。
    """
    super().__init__(**kwargs)    
    self.PER_e = 0.01  # 超参数，用于避免某些经验被选中的概率为0
    self.PER_a = 0.6  # 超参数，用于在只选高优先级经验和随机采样之间做权衡
    self.PER_b = 0.4  # 重要性采样，从初始值增加到1
    self.PER_BETA = 0.6 # 超参数，差值影响
    self.PER_b_increment_per_sampling = 0.001
    
    # self.absolute_error_upper = 1.  # 截断的绝对误差上限
    # ponytail: 原值1.0为Atari量纲(reward clip到[-1,1])的祖传超参；本环境|δ|合法范围~[0,140]
    # (碰撞终止步|δ|≈88~101)，原值使分子饱和、TD信号对采样分布贡献归零。取200覆盖合法支撑集上界。
    self.absolute_error_upper = 200.  # 截断的绝对误差上限
    self.tree = SumTree(self.capacity)
    return

    
  """
  在我们的树中存储新经验
  每个新经验的分数为max_priority（当我们用这个经验训练DDQN时会改进）
  """
  def store(self, experience):
    # 找到最大优先级
    max_priority = np.max(self.tree.tree[-self.tree.capacity:])
    
    # 如果最大优先级=0，我们不能设置优先级=0，因为这个经验永远不会被选中
    # 所以我们使用一个最小优先级
    if max_priority == 0:
        max_priority = self.absolute_error_upper
    
    self.tree.add(max_priority, experience)   # 为新经验设置最大优先级
    return
    
      
  """
  - 首先，要采样大小为k的小批量，将范围[0, priority_total]分成k个范围。
  - 然后从每个范围均匀采样一个值
  - 在sumtree中搜索，检索优先级分数对应采样值的经验。
  - 然后，计算每个小批量元素的IS权重
  """
  # def _sample_original(self, n):
  #   # 创建一个包含小批量的样本数组
  #   memory_buff = []
    
  #   b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
    
  #   # 计算优先级段
  #   # 这里，如论文所述，我们将范围[0, ptotal]分成n个范围
  #   priority_segment = self.tree.total_priority / n       # 优先级段

  #   # 这里每次采样新的小批量时增加PER_b
  #   self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # 最大=1
    
  #   # 计算max_weight，即最小优先级的权重（如果不太可能被采样到）
  #   p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
  #   max_weight = (p_min * n) ** (-self.PER_b)
    
  #   for i in range(n):
  #     """
  #     从每个范围均匀采样一个值
  #     """
  #     a, b = priority_segment * i, priority_segment * (i + 1)
  #     value = np.random.uniform(a, b)
      
  #     """
  #     检索对应每个值的经验
  #     """
  #     index, priority, data = self.tree.get_leaf(value)
      
  #     #P(j)
  #     sampling_proba = priority / self.tree.total_priority
      
  #     #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
  #     b_ISWeights[i, 0] = np.power(n * sampling_proba, -self.PER_b)/ max_weight
                             
  #     b_idx[i]= index
      
  #     experience = data #[data]
      
  #     memory_buff.append(experience)
  
  #   (states, actions, rewards, next_states, dones) = self._prepare_experience_buffer(memory_buff)
    
  #   if self.engine == 'torch':
  #     import torch as th
  #     b_ISWeights = th.from_numpy(b_ISWeights).float().to(self.device)
    
  #   return (states, actions, rewards, next_states, dones), b_idx, b_ISWeights


  def sample(self, n_samples):
    # 创建一个包含小批量的样本数组
        
    # 计算优先级段
    # 这里，如论文所述，我们将范围[0, ptotal]分成n_samples个范围
    priority_segment = self.tree.total_priority / n_samples       # 优先级段

    # 这里每次采样新的小批量时增加PER_b
    self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # 最大=1
    
    values = []
    for i in range(n_samples):
      """
      从每个范围均匀采样一个值
      """
      _a, _b = priority_segment * i, priority_segment * (i + 1)
      value = np.random.uniform(_a, _b)
      values.append(value)
    
    indices, priorities, datas = self.tree.get_leafs(values)
    
    # 不需要pow(p, alpha)，因为我们在更新时已经做了
    sampling_probas = priorities / self.tree.total_priority
    
    N = self.tree.stored_data
    
    weights = np.power(N * sampling_probas, -self.PER_b)
    max_weight = weights.max()
    
    np_IS_weights = weights / max_weight
  
    (states, actions, rewards, next_states, dones) = self._prepare_experience_buffer(datas)
    
    np_IS_weights = np_IS_weights.reshape((-1,1))
    
    if self.engine == 'torch':
      import torch as th
      out_IS_weights = th.from_numpy(np_IS_weights).float().to(self.device)
    else:
      out_IS_weights = np_IS_weights
    
    return (states, actions, rewards, next_states, dones), indices, out_IS_weights

  
  
  """
  更新树上的优先级
  """
  def batch_update(self, tree_idx, abs_errors, U):
    """
    更新树上的优先级（计算复合优先级）
    
    参数:
        tree_idx: 树节点索引
        abs_errors: TD误差绝对值
        U: 双Q值差异的不确定性指标
    
    复合优先级公式: P = |δ|^α / (1+U)^β
    """
    abs_errors += self.PER_e
    clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
    TD_error = clipped_errors
    cp = np.power(TD_error, self.PER_a) / np.power(1 + U, self.PER_BETA)

    for ti, p in zip(tree_idx, cp):
      self.tree.update(ti, p)    
      
      
  def __len__(self,):
    return min(self.tree.stored_data, self.tree.capacity)


PER = PERMemory
