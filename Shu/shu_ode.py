import torch
import dill
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
plt.rcParams['font.family'] = 'Times New Roman'
    
from src.ode_PINN_hardBC import ODE_PINN_HARDBC
from src.first_order_odesys_PINN_hardBC import ORDER1_ODESYS_PINN_HARDBC
from src.second_order_odesys_PINN_hardBC import ORDER2_ODESYS_PINN_HARDBC
from src.ode_PINN_softBC import ODE_PINN_SOFTBC
from src.ode_PINN_adaptCollectionPoint import  ODE_PINN_AdaptiveCollectionPoint

def Savepickle(obj, doc_path):
    with open(doc_path, 'wb') as file:
        dill.dump(obj, file)     

def Readpickle(doc_path):
    with open(doc_path, 'rb') as file:
        return dill.load(file)
    
def SaveModel(model, path): 
    if model.model_type == 'ODE_PINN_SOFTBC':
        data = {'model_type': model.model_type,
                'f': model.f, 
                'bound' : (model.lb,model.ub),
                'BC' : model.BC , 
                'lambdas' : model.lambdas,
                'n_hidden': model.n_hidden,
                'n_layers': model.n_layers,
                'rff_para': model.rff_para,
                'rff_B'   : model.rff_B,
                'para_dict': model.state_dict(), 
                'validate_loss': model.validate_loss,
                'train_loss': model.train_loss,
                'L2_loss': model.L2_loss}
    else:
        data = {'model_type': model.model_type,
                'f': model.f, 
                'bound' : (model.lb,model.ub),
                'BC' : model.BC , 
                'n_hidden': model.n_hidden,
                'n_layers': model.n_layers,
                'rff_para': model.rff_para,
                'rff_B'   : model.rff_B,
                'para_dict' : model.state_dict() , 
                'validate_loss' : model.validate_loss,
                'train_loss': model.train_loss,
                'L2_loss': model.L2_loss}
    Savepickle(data, path)
    
def LoadModel(path):
    data = Readpickle(path)
    
    if data['model_type'] == 'ODE_PINN_HARDBC':
        model = ODE_PINN_HARDBC(f=data['f'], lb=data['bound'][0], ub=data['bound'][1], BC=data['BC'], 
                                n_hidden=data['n_hidden'], n_layers=data['n_layers'])
        
    elif data['model_type'] == 'ODE_PINN_SOFTBC':
        model = ODE_PINN_SOFTBC(f=data['f'], lb=data['bound'][0], ub=data['bound'][1], BC=data['BC'], lambdas=data['lambdas'],
                                n_hidden=data['n_hidden'], n_layers=data['n_layers'])
        
    elif data['model_type'] == 'ODE_PINN_AdaptiveCollectionPoint':
        model = ODE_PINN_AdaptiveCollectionPoint(f=data['f'], lb=data['bound'][0], ub=data['bound'][1], BC=data['BC'], 
                                n_hidden=data['n_hidden'], n_layers=data['n_layers'])
        
    elif data['model_type'] == 'ORDER1_ODESYS_PINN_HARDBC':
        model = ORDER1_ODESYS_PINN_HARDBC(f=data['f'], lb=data['bound'][0], ub=data['bound'][1], BC=data['BC'], 
                                n_hidden=data['n_hidden'], n_layers=data['n_layers'])
        
    elif data['model_type'] == 'ORDER2_ODESYS_PINN_HARDBC':
        model = ORDER2_ODESYS_PINN_HARDBC(f=data['f'], lb=data['bound'][0], ub=data['bound'][1], BC=data['BC'], 
                                n_hidden=data['n_hidden'], n_layers=data['n_layers'])
     
    model.load_state_dict(data['para_dict'])
    model.validate_loss = data['validate_loss']
    model.train_loss = data['train_loss']
    model.L2_loss = data['L2_loss']
    return model

def moving_average(x, half_window_size):
    res = []
    x = np.array(x)
    for i in range(len(x)):
        if i == 0:
            res.append(x[i]) 
        elif i < half_window_size:
            res.append(x[0:2*i+1].mean())
        elif i <= len(x)-1-half_window_size:
            res.append(x[i-half_window_size:i+half_window_size+1].mean())
        else:
            res.append(x[2*i+1-len(x):].mean())
    return res


f = lambda x , y , D1y:  - (torch.pi)**2*y
true_sol = lambda x :  torch.sin(torch.pi*x)

solver = ODE_PINN_HARDBC( f=f, lb=-1, ub=1, BC=(2,0,-np.pi), n_hidden=16, n_layers=3)
solver.Train(train_num=3000, train_batch_size=64, learning_rate=0.01, 
             lr_step_size=100, min_lr =1e-10, lr_gamma=0.5,
             abs_tolerance=1e-4, max_epoch=5000, compute_L2_loss=True, true_sol=true_sol)
SaveModel(solver,'./data/ode_sin_slover_n1_3layer_16unit.pkl')

f = lambda x , y , D1y:  - (5*torch.pi)**2*y
true_sol = lambda x :  torch.sin(5*torch.pi*x)

solver = ODE_PINN_HARDBC( f=f, lb=-1, ub=1, BC=(2,0,-5*np.pi), n_hidden=16, n_layers=3)   ###
solver.Train(train_num=3000, train_batch_size=64, learning_rate=0.001, 
             lr_step_size=100, min_lr =1e-10, lr_gamma=0.5,
             abs_tolerance=1e-4, max_epoch=5000, compute_L2_loss=True, true_sol=true_sol)
SaveModel(solver,'./data/ode_sin_slover_n5_3layer_16unit.pkl')


solver = ODE_PINN_HARDBC( f=f, lb=-1, ub=1, BC=(2,0,-5*np.pi), n_hidden=16, n_layers=5)    ###
solver.Train(train_num=3000, train_batch_size=64, learning_rate=0.001, 
             lr_step_size=100, min_lr =1e-10, lr_gamma=0.5,
             abs_tolerance=1e-4, max_epoch=5000, compute_L2_loss=True, true_sol=true_sol)
SaveModel(solver,'./data/ode_sin_slover_n5_5layer_16unit.pkl')


solver = ODE_PINN_HARDBC( f=f, lb=-1, ub=1, BC=(2,0,-5*np.pi), n_hidden=32, n_layers=5)    ###
solver.Train(train_num=3000, train_batch_size=64, learning_rate=0.001, 
             lr_step_size=100, min_lr =1e-10, lr_gamma=0.5,
             abs_tolerance=1e-4, max_epoch=5000, compute_L2_loss=True, true_sol=true_sol)
SaveModel(solver,'./data/ode_sin_slover_n5_5layer_32unit.pkl')


solver = ODE_PINN_HARDBC( f=f, lb=-1, ub=1, BC=(2,0,-5*np.pi), n_hidden=64, n_layers=5)    
solver.Train(train_num=3000, train_batch_size=64, learning_rate=0.001, 
             lr_step_size=100, min_lr =1e-10, lr_gamma=0.5,
             abs_tolerance=1e-4, max_epoch=5000, compute_L2_loss=True, true_sol=true_sol)
SaveModel(solver,'./data/ode_sin_slover_n5_5layer_64unit.pkl')

 
solver = ODE_PINN_HARDBC( f=f, lb=-1, ub=1, BC=(2,0,-5*np.pi),                             ###
                         n_hidden=16, n_layers=3, set_rff=True, rff_num=8, u=0, std=1)
solver.Train(train_num=3000, train_batch_size=64, learning_rate=0.001, 
             lr_step_size=100, min_lr =1e-10, lr_gamma=0.5,
             abs_tolerance=1e-4, max_epoch=5000, compute_L2_loss=True, true_sol=true_sol)
SaveModel(solver,'./data/rff_ode_sin_slover_n5_3layer_16unit.pkl')


solver = ODE_PINN_AdaptiveCollectionPoint( f=f, lb=-1, ub=1, BC=(2,0,-5*np.pi), n_hidden=64, n_layers=5) ##
solver.Train(train_num=3000, train_batch_size=64, learning_rate=0.001, 
             lr_step_size=100, min_lr =1e-10, lr_gamma=0.5,
             abs_tolerance=1e-4, max_epoch=5000, compute_L2_loss=True, true_sol=true_sol)
SaveModel(solver,'./data/adp_ode_sin_slover_n5_5layer_64unit.pkl')


paths = [ './data/ode_sin_slover_n1_3layer_16unit.pkl',
          './data/ode_sin_slover_n5_3layer_16unit.pkl',
          './data/ode_sin_slover_n5_5layer_16unit.pkl',
          './data/ode_sin_slover_n5_5layer_32unit.pkl',
          './data/rff_ode_sin_slover_n5_3layer_16unit.pkl' ]

solvers = [LoadModel(path) for path in paths]


title_fontsize , tick_size , leg_size = 19, 17, 16
fig  = plt.figure(figsize = (11,5))
# color   = ['#2e2a79', '#473477', '#915270', '#C25371', '#E94220', '#6d6875']
# color   = ['#c1121f', '#802000','#003049','#386641','#6a4c93', '#6d6875']
color   = ['#000000', '#521400', '#802000', '#F53D00', '#FF7547', '#6d6875']
# color   = ["#0d0c1d","#161b33", "#474973", '#a69cac', '#f1dac4', '#6d6875']

legend  = ['REGPINN  n = 1\n3 layers  16 units',
           'REGPINN  n = 5\n3 layers  16 units',
           'REGPINN  n = 5\n5 layers  16 units',
           'REGPINN  n = 5\n5 layers  32 units',
           'RFFPINN   n = 5\n3 layers  16 units',]


ax_fit  = [plt.subplot(521), plt.subplot(523), plt.subplot(525), plt.subplot(527), plt.subplot(529)]
ax_loss = plt.subplot(122)

true_sol_1 = lambda x:  torch.sin(torch.pi*x)
true_sol_2 = lambda x:  torch.sin(5*torch.pi*x)
x = torch.linspace(-1,1,200).view(-1,1)
x_grid = x.squeeze().detach().numpy()

for i in range(5):
    if i == 0:
        y_gt = true_sol_1(x).squeeze().detach().numpy()
    else:
        y_gt = true_sol_2(x).squeeze().detach().numpy()
    y_hat = solvers[i](x).squeeze().detach().numpy()

    ax_fit[i].plot(x_grid, y_gt, color=color[-1], linewidth=2, linestyle='--')
    ax_fit[i].plot(x_grid, y_hat, color=color[i], linewidth=2, label=legend[i])
    ax_fit[i].grid()
    
    half_window_size = 10
    loss = np.log10(solvers[i].L2_loss)
    loss = moving_average(loss, half_window_size)
    ax_loss.plot(loss, color=color[i], linewidth=2, label=legend[i])
    
    ax_fit[i].tick_params(axis='x', labelsize=tick_size)
    ax_fit[i].tick_params(axis='y', labelsize=tick_size)
    
    if i == 4 :
        ax_fit[i].set_xlabel('x', fontsize=title_fontsize+5, fontweight='bold')
    ax_fit[i].set_ylabel('y', fontsize=title_fontsize, fontweight='bold', rotation=0)
    
ax_loss.grid()
ax_loss.set_ylim(-3, 1.5)
ax_loss.tick_params(axis='x', labelsize=tick_size)
ax_loss.tick_params(axis='y', labelsize=tick_size)

leg = ax_loss.legend(loc='center right', bbox_to_anchor=(1.63, 0.5), 
                     fontsize=leg_size, handlelength=1.0, labelspacing=1.4)

ax_loss.set_ylabel('Log10   L2   Loss', fontsize=title_fontsize, fontweight='bold')
ax_loss.set_xlabel('Epoch', fontsize=title_fontsize, fontweight='bold')

plt.subplots_adjust(wspace=0.25, hspace=0.7)
plt.show()