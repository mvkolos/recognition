import numpy as np
import torch
import torch.nn.init
import torch.nn as nn

gg = {}
gg['hook_position'] = 0
gg['total_fc_conv_layers'] = 0
gg['done_counter'] = -1
gg['hook'] = None
gg['act_dict'] = {}
gg['counter_to_apply_correction'] = 0
gg['correction_needed'] = False
gg['current_coef'] = 1.0

# Orthonorm init code is taked from Lasagne
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
def svd_orthonormal(w):
    shape = w.shape
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)#w;
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    print (shape, flat_shape)
    q = q.reshape(shape)
    return q.astype(np.float32)

def store_activations(self, input, output):
    gg['act_dict'] = output.data.cpu().numpy();
    #print('act shape = ', gg['act_dict'].shape)
    return


def add_current_hook(m):
    if gg['hook'] is not None:
        return
    if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        #print 'trying to hook to', m, gg['hook_position'], gg['done_counter']
        if gg['hook_position'] > gg['done_counter']:
            gg['hook'] = m.register_forward_hook(store_activations)
            #print ' hooking layer = ', gg['hook_position'], m
        else:
            #print m, 'already done, skipping'
            gg['hook_position'] += 1
    return

def count_conv_fc_layers(m):
    if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        gg['total_fc_conv_layers'] +=1
    return

def remove_hooks(hooks):
    for h in hooks:
        h.remove()
    return
def orthogonal_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if hasattr(m, 'weight_v'):
            w_ortho = svd_orthonormal(m.weight_v.data.cpu().numpy())
            m.weight_v.data = torch.from_numpy(w_ortho)
            try:
                nn.init.constant(m.bias, 0)
            except:
                pass
        else:
            #nn.init.orthogonal(m.weight)
            w_ortho = svd_orthonormal(m.weight.data.cpu().numpy())
            #print w_ortho 
            #m.weight.data.copy_(torch.from_numpy(w_ortho))
            m.weight.data = torch.from_numpy(w_ortho)
            try:
                nn.init.constant(m.bias, 0)
            except:
                pass
    return

def apply_weights_correction(m):
    if gg['hook'] is None:
        return
    if not gg['correction_needed']:
        return
    if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        if gg['counter_to_apply_correction'] < gg['hook_position']:
            gg['counter_to_apply_correction'] += 1
        else:
            if hasattr(m, 'weight_g'):
                m.weight_g.data *= float(gg['current_coef'])
                #print m.weight_g.data
                #print m.weight_v.data
                #print 'weights norm after = ', m.weight.data.norm()
                gg['correction_needed'] = False
            else:
                #print 'weights norm before = ', m.weight.data.norm()
                m.weight.data *= gg['current_coef']
                #print 'weights norm after = ', m.weight.data.norm()
                gg['correction_needed'] = False
            return
    return


def forward_model(model, data, device): 
    if isinstance(data, tuple):
        return model(*[x.to(device) for x in data])
    else:
        return model(data.to(device))


def LSUVinit(model, data, needed_std = 1.0, std_tol = 0.1, max_attempts = 10, do_orthonorm = True, device = 'cpu'):
    with torch.no_grad():
        model.eval();

        model = model.to(device)
        
        print ('Starting LSUV')
        model.apply(count_conv_fc_layers)
        
        print ('Total layers to process:', gg['total_fc_conv_layers'])
        if do_orthonorm:
            model.apply(orthogonal_weights_init)
            print ('Orthonorm done')
            model = model.to(device)

        for layer_idx in range(gg['total_fc_conv_layers']):
            print (layer_idx)
            model.apply(add_current_hook)
            
            out = forward_model(model, data, device)

            current_std = gg['act_dict'].std()
            print ('std at layer ',layer_idx, ' = ', current_std)

            attempts = 0
            while (np.abs(current_std - needed_std) > std_tol):
                gg['current_coef'] =  needed_std / (current_std  + 1e-8);
                gg['correction_needed'] = True
                model.apply(apply_weights_correction)
                
                model = model.to(device)

                out = forward_model(model, data, device)

                current_std = gg['act_dict'].std()
                print ('std at layer ',layer_idx, ' = ', current_std, 'mean = ', gg['act_dict'].mean())
                attempts+=1
                if attempts > max_attempts:
                    print ('Cannot converge in ', max_attempts, 'iterations')
                    break
            
            if gg['hook'] is not None:
               gg['hook'].remove()
            
            gg['done_counter']+=1
            gg['counter_to_apply_correction'] = 0
            gg['hook_position'] = 0
            gg['hook']  = None
            
            print ('finish at layer',layer_idx)
        
        print ('LSUV init done!')
        model = model.to(device)
        
        return model
