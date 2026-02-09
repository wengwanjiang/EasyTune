import random
from omegaconf import OmegaConf
random.seed(42)
#lambda_reward = '1e0'

def float_to_1ex(x):
    if x == 0:
        return "0e0"
    s = f"{x:.0e}"  # 科学计数法格式化，如 "1e-01"
    return s.replace('+', '')

def ft_reset(ft_config):
    ft_type = ft_config['type']
    lambda_reward_1ex = float_to_1ex(ft_config['lambda_reward'])
    assert ft_type in ['ReFL', 'DRaFT', 'DRTune', 'AlignProp', 'EZtune'], f"Invalid type: {ft_type}"
    if ft_type == 'EZtune':
        k = ft_config['k']
        ft_config['enable_grad'] = [i for i in range(50)][-k:]
        return ft_config
    if ft_type == 'ReFL':
        ft_config['t_min'] = random.randint(50 - ft_config['M'] + 1, 50 - 1) # randint [a, b] is inclusive
        ft_config['enable_grad'] = [x for x in range(ft_config['t_min'], 50)]
        return ft_config
    elif ft_type == 'AlignProp':
        while True:
            enable_grad = [i for i in range(50) if random.random() < ft_config['prob']]
            if len(enable_grad) > 0:
                ft_config['enable_grad'] = enable_grad
                return ft_config
    elif ft_type == 'DRaFT':
        k = ft_config['k']
        if ft_config['custom'] is not None:
            ft_config['enable_grad'] = ft_config['custom']
            custom_str = '_['
            for i in ft_config['custom']:
                custom_str += str(i) + ','
            custom_str += + ']_'
            ft_config['name'] = f"DRaFT_C{custom_str}_R{lambda_reward_1ex}"
            
            return ft_config
        ft_config['name'] = f"DRaFT_K{k}_G{int(ft_config['skip'])}_Re{int(ft_config['reverse'])}_R{lambda_reward_1ex}"
        if ft_config['skip'] == True and ft_config['reverse'] == False:
            ft_config['enable_grad'] = [i for i in range(50)][::k]
            return ft_config
        elif ft_config['skip'] == False and ft_config['reverse'] == False:
            ft_config['enable_grad'] = [i for i in range(50)][-k:]
            ft_config['t_train'] = ft_config['enable_grad']
            return ft_config
        elif ft_config['skip'] == False and ft_config['reverse'] == True:
            ft_config['enable_grad'] = [i for i in range(50)][:k]
            return ft_config
        elif ft_config['skip'] == True and ft_config['reverse'] == True:
            ft_config['enable_grad'] = [i for i in range(50)][::-k]
            return ft_config
    elif ft_type == 'DRTune':
        ft_config['s'] = random.randint(0, ft_config['T'] - (ft_config['T'] // ft_config['k']) * ft_config['k'])
        ft_config['t_train'] = [ft_config['s'] + i * (ft_config['T'] // ft_config['k']) for i in range(ft_config['k'])]
        ft_config['t_min'] = random.randint(ft_config['T'] - ft_config['M'] + 1, 50 - 1) # randint [a, b] is inclusive
        ft_config['enable_grad'] = [i for i in range(50)]
        return ft_config
    
def get_ft_config(ft_type, m=1, prob=0.1, t=50, k=1, skip=False, reverse=False, custom=None, lambda_reward=1e0, dy=None):
    """
    Get the finetune config based on the type of finetune.
    Args:
        ft_type (str): The type of finetune. Can be 'ReFL', 'DRaFT', 'DRTune', 'AlignProp'.
        m (int): The number of steps to finetune.
        prob (float): The probability of selecting a gradient.
        t (int): The number of steps to train.
        k (int): The number of steps to skip.
        skip (bool): Whether to skip the first k steps.
        reverse (bool): Whether to reverse the order of the gradients.
        custom (list): A list of custom gradients to select.
        lambda_reward (float): The Weight of lambda reward.
    """
    assert ft_type in ['ReFL', 'DRaFT', 'DRTune', 'AlignProp', 'EZtune'], f"Invalid type: {ft_type}"
    if ft_type == 'ReFL':
        return ReFLClass(m, lambda_reward=lambda_reward)
    elif ft_type == 'AlignProp':
        return AlignPropClass(prob, lambda_reward=lambda_reward)
    elif ft_type == 'DRaFT':
        return DRaFTClass(k=k, reverse=reverse, skip=skip, custom=custom, lambda_reward=lambda_reward)
    elif ft_type == 'DRTune':
        return DRTuneClass(m=m, t=t, k=k, lambda_reward=lambda_reward)
    elif ft_type == 'EZtune':
        return EZtuneClass(k=k, lambda_reward=lambda_reward, dy=dy)
    
def EZtuneClass(k, lambda_reward, dy=None):
    EZtune = {}
    EZtune['type'] = 'EZtune'
    EZtune['k'] = k
    lambda_reward_1ex = float_to_1ex(lambda_reward)
    EZtune['name'] = f"EZtune_K{k}_R{lambda_reward_1ex}"
    EZtune['lambda_reward'] = float(lambda_reward)
    EZtune['dy'] = dy
    EZtune = ft_reset(EZtune)
    return EZtune



def AlignPropClass(prob, lambda_reward):
    AlignProp = {}
    AlignProp['type'] = 'AlignProp'
    AlignProp['prob'] = prob
    lambda_reward_1ex = float_to_1ex(lambda_reward)
    AlignProp['name'] = f"AlignProp_P{int(prob*100)}_R{lambda_reward_1ex}"
    if prob < 1e-3:
        assert 1 == 2, 'prob < 1e-3 is not supported'
    AlignProp['lambda_reward'] = float(lambda_reward)
    AlignProp = ft_reset(AlignProp)
    return AlignProp
        

def DRaFTClass(k, reverse=False, skip=False, custom=None, lambda_reward=None, maxT=50):
    DRaFT = {}
    DRaFT['type'] = 'DRaFT'
    DRaFT['k'] = k
    DRaFT['maxT'] = maxT
    DRaFT['skip'] = skip
    DRaFT['reverse'] = reverse
    DRaFT['custom'] = custom
    DRaFT['lambda_reward'] = float(lambda_reward)
    DRaFT = ft_reset(DRaFT)  
    return DRaFT
        
def DRTuneClass(m, t, k, lambda_reward, maxT=50):
    DRTune = {}
    DRTune['type'] = 'DRTune'
    DRTune['M'] = m
    DRTune['t_min'] = random.randint(maxT - DRTune['M'] + 1, 50 - 1) # randint [a, b] is inclusive
    DRTune['T'] = t
    DRTune['k'] = k
    DRTune['lambda_reward'] = float(lambda_reward)
    lambda_reward_1ex = float_to_1ex(lambda_reward)
    DRTune['name'] = f"DRTune_T{DRTune['T']}_K{DRTune['k']}_M{DRTune['t_min']}_R{lambda_reward_1ex}"
    DRTune = ft_reset(DRTune)
    return DRTune
    
def ReFLClass(m, lambda_reward):
    ReFL = {}
    ReFL['type'] = 'ReFL'
    ReFL['M'] = m
    lambda_reward_1ex = float_to_1ex(lambda_reward)
    ReFL['name'] = f"ReFL_M{ReFL['M']}_R{lambda_reward_1ex}"
    ReFL['lambda_reward'] = float(lambda_reward)
    ReFL = ft_reset(ReFL)
    return ReFL
    
        
        


