from turtle import width
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from scipy.fft import fft
from dtft_helper import *
import params
import random
# plt.locator_params(axis='x', nbins=4) THIS BROKE THE ANIMATION CODE

def split_data(dataset,bsize,remove_rear=100,train_size=.7,val_size=None):
    # seed = params.seed_val
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)


    # np.random.seed(111)
    sheet_count = len(dataset)
    split = int(np.floor(train_size * sheet_count))
    # print("TRAIN TEST SPPLIT POINT",split,sheet_count)
    indices = list(range(1,sheet_count-1)) #remove first and last
    np.random.shuffle(indices)
    # print("sheet indices",indices,"split point",split)
    train_indices, val_indices = indices[:split], indices[split:]
    # print(len(train_indices),len(val_indices),val_indices)
    # train_dataset = Subset(dataset, train_indices)
    # train_loader = DataLoader(train_indices,  shuffle=True)
    if not val_size:
        # test_loader = DataLoader(val_indices, batch_size=1, shuffle=False)
        # return train_loader,test_loader
        return train_indices,val_indices



    test_sheet_count = len(val_indices)
    val_split = int(np.floor(val_size * test_sheet_count))
    np.random.shuffle(val_indices)
    test_indices,validation_indices =  val_indices[:val_split], val_indices[val_split:]
    # test_loader = DataLoader(test_indices, batch_size=1, shuffle=False)
    # val_loader = DataLoader(validation_indices, batch_size=1, shuffle=False)
    # print("Val TEST SPPLIT POINT",len(test_indices),len(validation_indices))


    # test_dataset = Subset(dataset, val_indices)

    # fixed_offset_dataset = Subset(dataset, list(range(len(dataset)-100,len(dataset))))
    # fixed_offset_loader = DataLoader(fixed_offset_dataset, batch_size=1, shuffle=False)

    return train_indices,validation_indices,test_indices

# def dtft(data):
#     return fft(data)

def tranform_batch_BIN(train_data_batch,n_objects,object_dim,USE_CUDA,noise=1):
    batch_size = len(train_data_batch['landmarks'])
    data_time = train_data_batch['time']
    landmarks = train_data_batch['landmarks']
    output = train_data_batch['output']
    # import pdb;pdb.set_trace()
    obj_attr = landmarks[:,:n_objects*object_dim]
    # noise=1
    # objects = np.reshape(obj_attr,(batch_size,n_objects,object_dim)) +np.concatenate( (noise*np.random.normal(0,.05,(batch_size,n_objects,2)) , noise*np.random.normal(0,.005,(batch_size,n_objects,2))), axis =2)
    objects = np.reshape(obj_attr,(batch_size,n_objects,object_dim)) +np.concatenate( (noise*np.random.normal(0,.05,(batch_size,n_objects,2)) , noise*np.random.normal(0,.005,(batch_size,n_objects,2))), axis =2)

    # print(len(objects))
    objects = np.asarray(objects, dtype = np.float64, )
  
    sender_relations = np.repeat(params.sender_rel[None, :, :], batch_size, axis=0)
    receiver_relations = np.repeat(params.receiver_rel[None, :, :], batch_size, axis=0)

# SANITY CHECK
    # sender_ids = np.argmax(params.sender_rel, axis=0)
    # receiver_ids = np.argmax(params.receiver_rel, axis=0)
    # print(sender_ids,receiver_ids)
    rel_id = np.reshape(np.tile((1,-1),np.shape(params.sender_rel)[-1]//2),(1,1,-1))
    rel_id = np.repeat(rel_id,repeats=batch_size,axis=0)
    rel_id= np.squeeze(rel_id,axis=1)

    coords_ids = params.coords_ids
    sender_ids = np.argmax(params.sender_rel, axis=0)
    receiver_ids = np.argmax(params.receiver_rel, axis=0)
    # print("UTILS rep\n",sender_ids,"\n",receiver_ids)
    dist_cols=[]
    angles_cols=[]

    for s_id,r_id in zip(sender_ids,receiver_ids):
        # print("utisl rep",s_id,r_id)
        # print("utislrep",str(coords_ids[s_id]),str(coords_ids[r_id]))
        dist_cols.append("rx"+str(coords_ids[s_id])+"x"+str(coords_ids[r_id]))
        angles_cols.append("@x"+str(coords_ids[s_id])+"x"+str(coords_ids[r_id]))

    # print("utils rep",train_data_batch['col_id'])
    # import pdb;pdb.set_trace()
    dist_cols = [list(train_data_batch['col_id']).index(v) for v in dist_cols]
    angles_cols = [list(train_data_batch['col_id']).index(v) for v in angles_cols]

    # import pdb;pdb.set_trace()
    dist_rel_attr = landmarks[:,dist_cols]
    theta_rel_attr= landmarks[:,angles_cols]
    # for id in landmark_rel_index[1:]:
    #     dist_rel_attr = np.concatenate((dist_rel_attr,landmarks[:,id:id+1]),axis=1)
    #     theta_rel_attr =np.concatenate((theta_rel_attr,landmarks[:,id+1:id+2]),axis=1)
    # joints_ids = ['12x14','']
    
    # for coord in coords_id:
    #     dist_cols.append("dx")
    #     angles_cols.append("@x")

    # landmark_rel_index = [20,28,22,36,30,38,46,32,56,42]
    
    # dist_rel_attr = landmarks[:,20:21]
    # theta_rel_attr= landmarks[:,21:22]
    # for id in landmark_rel_index[1:]:
    #     dist_rel_attr = np.concatenate((dist_rel_attr,landmarks[:,id:id+1]),axis=1)
    #     theta_rel_attr =np.concatenate((theta_rel_attr,landmarks[:,id+1:id+2]),axis=1)
    # print("shape rel matrix",np.shape(rel_id),np.shape(theta_rel_attr))
    # import pdb;pdb.set_trace()
    rel_matrix = np.stack([rel_id,dist_rel_attr,theta_rel_attr],axis=2)
    # print(np.shape(rel_matrix))
    # rel_matrix = np.zeros((batch_size,params.n_relations,3))
    # rel_matrix = np.squeeze(rel_matrix)
# 20 NL 1st is reeiver 2nd is sender
# 22 NR
# 28 lN
# 30 LR
# 32 LLK
# 36 RN
# 38 RL
# 42 RRK
# 46 LKL
# 56 RKR
    external_effect_info = np.repeat([[[.98]]],n_objects,axis=1)
    external_effect_info = np.repeat(external_effect_info,batch_size,axis=0)

    target_attr = output[:,:n_objects*object_dim]
    # print(np.shape(target_attr))
    target = np.reshape(target_attr,(batch_size,n_objects,object_dim))
    # target = target.float()
    target = np.asarray(target, dtype = np.float64, )


    # print(np.shape)
    objects = Variable(torch.FloatTensor(objects))
    sender_relations   = Variable(torch.FloatTensor(sender_relations))
    receiver_relations = Variable(torch.FloatTensor(receiver_relations))
    rel_matrix      = Variable(torch.FloatTensor(rel_matrix))
    external_effect_info = Variable(torch.FloatTensor(external_effect_info))
    target             = Variable(torch.FloatTensor(target)).reshape(-1, object_dim)# 4 size output
                       
    if USE_CUDA:
        objects            = objects.cuda()
        sender_relations   = sender_relations.cuda()
        receiver_relations = receiver_relations.cuda()
        rel_matrix      = rel_matrix.cuda()
        external_effect_info = external_effect_info.cuda()
        target             = target.cuda()
    
    return objects, sender_relations, receiver_relations, rel_matrix, external_effect_info, target


def get_landmark_errors(target,prediction,loss_id):
    #     loss_id = {'nose':[],'lh':[],'rh':[],'lk':[],'rk':[],'total':[]}
    ct = 0
    while ct < len(target):
        for key in loss_id.keys():
            # print(len(target),ct)
            # import pdb;pdb.set_trace(   )
            if key == 'total':# and ct == 5:
                # loss_id[key].append(torch.sum(torch.abs(torch.subtract(target,prediction))).item())
                loss_id[key].append(np.sum([loss_id[key][-1] for key in loss_id.keys() if key !='total']))
                break
            loss_id[key].append(torch.sum(torch.abs(torch.subtract(target[ct],prediction[ct]))).item())
            ct+=1
            # ct=ct%5
    return loss_id

def plot_loss(loss,save_path,dataset_type,loss_id=None,body=False):
    if not body:
        fig, axs = plt.subplots(1, 1)
        axs.plot(loss)
        axs.set_title('MSE '+str(np.round(np.mean(loss),2)))
    else:
        width = 2
        height = len(params.coords_ids)//width +1  # oneextra for total
        fig, axs = plt.subplots(height, width,figsize=(15,15))
        for id,key in enumerate(params.keys):
            # axs[id//width, id%width].plot(loss)
            # axs[id//width, id%width].set_title('MSE '+str(np.round(np.mean(loss),2)))
            axs[id//width, id%width].plot(loss_id[params.keys[id]])
            axs[id//width, id%width].set_title('Loss_'+params.keys[id]+": "+str(np.round(np.mean(loss_id[params.keys[id]]),2)))
        axs[height-1, width-1].plot(loss_id['total'])
        axs[height-1, width-1].set_title('Loss_Total: '+str(np.round(np.mean(loss_id['total']),2)))
        # id = 0
        # axs[0, 0].plot(loss)
        # axs[0, 0].set_title('MSE '+str(np.round(np.mean(loss),2)))
        # axs[0, 1].plot(loss_id[params.keys[id]])
        # axs[0, 1].set_title('Abs Loss: '+params.keys[id]+str(np.round(np.mean(loss_id[params.keys[id]]),2)))
        # id+=1
        # axs[1, 0].plot(loss_id[params.keys[id]])
        # axs[1, 0].set_title('Abs Loss: '+params.keys[id]+str(np.round(np.mean(loss_id[params.keys[id]]),2)))
        # id+=1

        # axs[1, 1].plot(loss_id[params.keys[id]])
        # axs[1, 1].set_title('Abs Loss: '+params.keys[id]+str(np.round(np.mean(loss_id[params.keys[id]]),2)))
        # id+=1


        # axs[2, 0].plot(loss_id[params.keys[id]])
        # axs[2, 0].set_title('Abs Loss:'+params.keys[id]+str(np.round(np.mean(loss_id[params.keys[id]]),2)))
        # id+=1

        # axs[2, 1].plot(loss_id[params.keys[id]])
        # axs[2, 1].set_title('Abs Loss:'+params.keys[id]+str(np.round(np.mean(loss_id[params.keys[id]]),2)))
        id =0
        for ax in axs.flat:
            id+=1
            ax.set(xlabel='Time', ylabel='Error')
            tick_spacing=50
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            if id < height*width-1:
                ax.set_xticks([])
                ax.set(xlabel='')

    # plt.show()
    plt.savefig(save_path+ dataset_type+ ".jpg")
    plt.close()

def create_subplots():
    ul=.3
    ul_t=.5
    width = 2
    height = len(params.coords_ids)//width +1  # oneextra for total
    ax1 = plt.subplot2grid((height, 2*width), (0, 0),rowspan=height,colspan=width)
    ax1.title.set_text("IN")
    ax1.set_xlim([-.20,1.4])
    ax1.set_ylim([-0.65,1])
    ax1.grid()

    ax_o = []
    for id,key in enumerate(params.keys):
        ax_ = plt.subplot2grid((height, 2*width), (id//width, width+id%width))
        ax_.title.set_text("MSE " + params.keys[id])
        ax_.set_xlim([-1,501])
        ax_.set_ylim([0,ul])
        ax_.grid()
        ax_o.append(ax_)
        # axs[id//width, id%width].plot(loss)
        # axs[id//width, id%width].set_title('MSE '+str(np.round(np.mean(loss),2)))
        # axs[id//width, id%width].plot(loss_id[params.keys[id]])
        # axs[id//width, id%width].set_title('Loss_'+params.keys[id]+": "+str(np.round(np.mean(loss_id[params.keys[id]]),2)))
    ax_total = plt.subplot2grid((height, 2*width), (height-1, 2*width-1))
    ax_total.title.set_text("TOTAL MSE " )
    ax_total.set_xlim([-1,501])
    ax_total.set_ylim([0,ul_t])
    ax_total.grid()
   
    return ax1,ax_o,ax_total

def stick_animation(loss_id,ground_truth,predictions,name="Basic",xshift=0.1,savepath=None):
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(27, 15))
    dx,dy = 0,1
    ax1,ax_o,ax_total= create_subplots()
    
    gt_line, = ax1.plot([], [], ms=6, c= 'r')
    pd_line, = ax1.plot([], [], ms=6, c= 'b')
    
    errors=[]
    vals=[]
    for axs in ax_o:
        error_k, = axs.plot([], [],ms=6)
        val_k = axs.text(380,0.2,"",fontsize=25)
        errors.append(error_k)
        vals.append(val_k)
    error_total, = ax_total.plot([], [],ms=6)
    val_total = ax_total.text(380,0.2,"",fontsize=25)
    # error_n, = ax3.plot([], [],ms=6)
    # val_n = ax3.text(380,0.1,"",fontsize=25)

    # error_rh, = ax4.plot([], [],ms=6)
    # val_rh = ax4.text(380,0.1,"",fontsize=25)
    # error_lh, = ax5.plot([], [],ms=6)
    # val_lh = ax5.text(380,0.1,"",fontsize=25)
    # error_rk, = ax6.plot([], [],ms=6)
    # val_rk = ax6.text(380,0.1,"",fontsize=25)
    # error_lk, = ax7.plot([], [],ms=6)
    # val_lk = ax7.text(380,0.1,"",fontsize=25)
    
    def set_stick_model(pos,ax,xshift=0):
        coords_ids = params.coords_ids
        sender_ids = np.argmax(params.sender_rel, axis=0)
        receiver_ids = np.argmax(params.receiver_rel, axis=0)
        positions = []
        # print(pos)
        # print(pos[1,0])
        for s_id,r_id in zip(sender_ids[::2],receiver_ids[::2]):
            positions.extend([pos[s_id,ax].item(),pos[r_id,ax].item(),None])
            # print("Sender id %d, receiver id %d"%(s_id,r_id) , [pos[s_id,ax].item(),pos[r_id,ax].item(),None], ax)

        # positions= [pos[0,ax].item(),pos[1,ax].item(), None,pos[1,ax].item(),pos[2,ax].item(),None, \
        #         pos[0,ax].item(),pos[3,ax].item(),None, pos[0,ax].item(),pos[4,ax].item(),None,\
        #         pos[3,ax].item(),pos[5,ax].item(),None]
        positions=[xshift+i if i is not None else i for i in positions]
        return positions
    # initialization function: plot the background of each frame
    def init():
        gt_line.set_data([], [])
        pd_line.set_data([], [])
        for error_k, val_k in zip(errors,vals):
            error_k.set_data([],[])
            val_k.set_text('')
        # print(gt_line)
        # error_total = error_total[0] # dont know why this is a list instead of line2d object
        error_total.set_data([],[])
        val_total.set_text('')


        return gt_line,pd_line,errors,vals,error_total,val_total,#error_t,val_t,error_n,val_n,error_rh,val_rh,error_lh,val_lh,error_rk,val_rk,error_lk,val_lk,
    
    def animate(i):

        # print(i,"nose x",ground_truth[i][0,0])
        x = set_stick_model(ground_truth[i],dx)
        y = set_stick_model(ground_truth[i],dy)#np.abs(ground_truth[i]-1),dy)
        gt_line.set_data(x,y)  # update the data.
        # print(x,y)
        x = set_stick_model(predictions[i],dx,xshift)
        y = set_stick_model(predictions[i],dy)#np.abs(predictions[i]-1),dy) # this is done to reverse the diretion of y axis, now y values decrease from top to bottom
        pd_line.set_data(x,y)  # update the data.        
        # print(loss,"offshoot",x,y)

        x_er = [k for k in range(i)]
        id=0
        for error_k,val_k in zip(errors,vals):
            error_k.set_data(x_er,loss_id[params.keys[id]][:i])
            val_k.set_text("{:.5f}".format(loss_id[params.keys[id]][i]))
            id+=1
        

        error_total.set_data(x_er, loss_id['total'][:i])
        val_total.set_text("{:.5f}".format( loss_id['total'][i]))
        return gt_line,pd_line,errors,vals,error_total,val_total,#,error_n,val_n,error_rh,val_rh,error_lh,val_lh,error_rk,val_rk,error_lk,val_lk,
    
    ani = animation.FuncAnimation(
        plt.figure(1), animate, init_func=init,frames=int(1*len(loss_id['total']))-5, interval=40, blit=False, save_count=50)
    # plt.show()

    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    if savepath is None:
        savepath = "results/"+input_data_file+"/"
    ani.save(savepath+name+".mp4", writer=writer) # https://www.pythonfixing.com/2022/02/fixed-how-to-speed-up-animation-in.html for speed
    plt.close()
    return ani

def normalize_time(time_values,future_offset):#should be train data
    # import pdb;pdb.set_trace()
    time_durations = []
    for t in range(1,future_offset+1):
        temp = [ (time_values[i+t]-time_values[i])[0] for i in range(len(time_values)-t)]
        time_durations = time_durations + temp
#        
    return min(time_durations),max(time_durations)

def tranform_batch_TINE(batch_size,train_data_batch,n_objects,object_dim,USE_CUDA, MIN_T, MAX_T, n_relations, future_offset ,noise=1):
    batch_size = len(train_data_batch['landmarks'])  # REDUCED 1 coz we dont have output time info for last entry of batch input
    # import pdb;pdb.set_trace()
    data_time = train_data_batch['time']
    landmarks = train_data_batch['landmarks']
    output = train_data_batch['output']
    # import pdb;pdb.set_trace()
    obj_attr = landmarks[:batch_size,:20]
    # noise=1
    objects = np.reshape(obj_attr,(batch_size,n_objects,object_dim)) +np.concatenate( (noise*np.random.normal(0,.001,(batch_size,n_objects,2)) , noise*np.random.normal(0,.0001,(batch_size,n_objects,2))), axis =2)

    objects =  np.asarray(objects, dtype = np.float64, )
  

    if n_objects == 5:
        receiver_rel =[
                [1,0,1,0,0,0,0,0,0,0],
                [0,1,0,0,1,0,0,1,0,0],
                [0,0,0,1,0,1,0,0,0,1],
                [0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                
                ]
        receiver_relations = np.broadcast_to(receiver_rel,(batch_size,)+np.shape(receiver_rel))


        sender_rel =  [
                [0,1,0,1,0,0,0,0,0,0],
                [1,0,0,0,0,1,1,0,0,0],
                [0,0,1,0,1,0,0,0,1,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,0,1],
                
                ]
        sender_relations = np.broadcast_to(sender_rel,(batch_size,)+np.shape(sender_rel))
    elif n_objects==7:
        receiver_rel =[
                    [1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
                    [0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0],
                    [0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
                    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],
                    ]
        receiver_relations = np.repeat(receiver_rel[None, :, :], batch_size, axis=0)

        sender_rel =  [
                    [0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0],
                    [1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],
                    [0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0],
                    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
                    ]
        sender_relations = np.repeat(sender_rel[None, :, :], batch_size, axis=0)

    rel_id = np.reshape([1,-1,1,-1,1,-1,1,-1,1,-1],(1,1,-1))
    rel_id = np.repeat(rel_id,repeats=batch_size,axis=0)
    rel_id=np.squeeze(rel_id,axis=1)
    landmark_rel_index = [20,28,22,36,30,38,46,32,56,42]
    dist_rel_attr = landmarks[:batch_size,20:21]
    theta_rel_attr= landmarks[:batch_size,21:22]
    for id in landmark_rel_index[1:]:
        dist_rel_attr = np.concatenate((dist_rel_attr,landmarks[:batch_size,id:id+1]),axis=1)
        theta_rel_attr =np.concatenate((theta_rel_attr,landmarks[:batch_size,id+1:id+2]),axis=1)
  
    # print("shape rel matrix",np.shape(rel_id),np.shape(theta_rel_attr))
    rel_matrix = np.stack([rel_id,dist_rel_attr,theta_rel_attr],axis=2)
    # rel_matrix = np.zeros((batch_size,10,3))
    # rel_matrix = np.squeeze(rel_matrix)
# 20 NL 1st is reeiver 2nd is sender
# 22 NR
# 28 lN
# 30 LR
# 32 LLK
# 36 RN
# 38 RL
# 42 RRK
# 46 LKL
# 56 RKR
    external_effect_info = np.repeat([[[.98]]],n_objects,axis=1)
    external_effect_info = np.repeat(external_effect_info,batch_size,axis=0)
    
    output_index = [np.random.randint(i,min(i+future_offset,batch_size) ) for i in range(batch_size)]
    if batch_size ==1: # for testing case
        output_index = [min(i+future_offset,batch_size) for i in range(batch_size)]
    
    target_attr = output[:batch_size,:20]
    # print(np.shape(target_attr))
    target = np.reshape(target_attr,(batch_size,n_objects,object_dim))
    target =  np.asarray(target, dtype = np.float64, )

    # print(np.shape(data_time))
    # import pdb;pdb.set_trace()
    # data_time= np.squeeze(data_time)
    # output_index = [1+v for v in output_index]
    # duration_data = np.subtract(data_time[output_index,:] , data_time[:batch_size,:])

    duration_data_normalised = (data_time - MIN_T)/(MAX_T-MIN_T)# (duration_data - MIN_T)/(MAX_T-MIN_T)
    # time_durations = np.repeat(duration_data_normalised, n_relations,axis=1)# num relations is 16 but for input model i need dim = num objects
    duration_data_normalised = np.expand_dims(duration_data_normalised,-1)
    time_durations= np.asarray(duration_data_normalised, dtype = np.float64, ) 
    # print("time duration",np.shape(time_durations),time_durations,MIN_T,MAX_T)

    objects = Variable(torch.FloatTensor(objects))
    time_durations       = Variable(torch.FloatTensor(time_durations))
    sender_relations   = Variable(torch.FloatTensor(sender_relations))
    receiver_relations = Variable(torch.FloatTensor(receiver_relations))
    rel_matrix      = Variable(torch.FloatTensor(rel_matrix))
    external_effect_info = Variable(torch.FloatTensor(external_effect_info))
    target             = Variable(torch.FloatTensor(target)).reshape(-1, object_dim)# 4 size output
                       
    if USE_CUDA:
        objects            = objects.cuda()
        time_durations       = time_durations.cuda()

        sender_relations   = sender_relations.cuda()
        receiver_relations = receiver_relations.cuda()
        rel_matrix      = rel_matrix.cuda()
        external_effect_info = external_effect_info.cuda()
        target             = target.cuda()
    
    return objects, time_durations, sender_relations, receiver_relations, rel_matrix, external_effect_info, target


def transform_batch_TINI(batch_size,train_data_batch,n_objects,object_dim,USE_CUDA, MIN_T, MAX_T, n_relations, future_offset ,noise=1):
    # batch_size = len(train_data_batch['landmarks']) - 1 # REDUCED 1 coz we dont have output time info for last entry of batch input
    data_time = train_data_batch['time']
    landmarks = train_data_batch['landmarks']
    output = train_data_batch['output']
    obj_attr = landmarks[:batch_size,:,:20]
    # noise=1
    objects = np.reshape(obj_attr,(batch_size,n_objects,object_dim)) +np.concatenate( (noise*np.random.normal(0,.04,(batch_size,n_objects,2)) , noise*np.random.normal(0,.0004,(batch_size,n_objects,2))), axis =2)
    objects = objects.float()
  

    if n_objects == 5:
        receiver_rel =[
                [1,0,1,0,0,0,0,0,0,0],
                [0,1,0,0,1,0,0,1,0,0],
                [0,0,0,1,0,1,0,0,0,1],
                [0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                
                ]
        receiver_relations = np.broadcast_to(receiver_rel,(batch_size,)+np.shape(receiver_rel))


        sender_rel =  [
                [0,1,0,1,0,0,0,0,0,0],
                [1,0,0,0,0,1,1,0,0,0],
                [0,0,1,0,1,0,0,0,1,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,0,1],
                
                ]
        sender_relations = np.broadcast_to(sender_rel,(batch_size,)+np.shape(sender_rel))

    else:
        receiver_rel =[
                    [1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
                    [0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0],
                    [0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
                    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],
                    ]
        receiver_relations = np.repeat(receiver_rel[None, :, :], batch_size, axis=0)

        sender_rel =  [
                    [0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0],
                    [1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],
                    [0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0],
                    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
                    ]
        sender_relations = np.repeat(sender_rel[None, :, :], batch_size, axis=0)

    rel_id = np.reshape([1,-1,1,-1,1,-1,1,-1,1,-1],(1,1,-1))
    rel_id = np.repeat(rel_id,repeats=batch_size,axis=0)
    rel_id=np.squeeze(rel_id,axis=1)
    landmark_rel_index = [20,28,22,36,30,38,46,32,56,42]
    dist_rel_attr = landmarks[:batch_size,:,20]
    theta_rel_attr= landmarks[:batch_size,:,21]
    for id in landmark_rel_index[1:]:
        dist_rel_attr = np.concatenate((dist_rel_attr,landmarks[:batch_size,:,id]),axis=1)
        theta_rel_attr =np.concatenate((theta_rel_attr,landmarks[:batch_size,:,id+1]),axis=1)
    # print("shape rel matrix",np.shape(rel_id),np.shape(theta_rel_attr))
    rel_matrix = np.stack([rel_id,dist_rel_attr,theta_rel_attr],axis=2)
    # rel_matrix = np.zeros((batch_size,10,3))
    # rel_matrix = np.squeeze(rel_matrix)
# 20 NL 1st is reeiver 2nd is sender
# 22 NR
# 28 lN
# 30 LR
# 32 LLK
# 36 RN
# 38 RL
# 42 RRK
# 46 LKL
# 56 RKR
    external_effect_info = np.repeat([[[.98]]],n_objects,axis=1)
    external_effect_info = np.repeat(external_effect_info,batch_size,axis=0)
    
    # output_index = [np.random.randint(i,min(i+future_offset,batch_size) ) for i in range(batch_size)]
    # if batch_size ==1: # for testing case
    #     output_index = [min(i+future_offset,batch_size) for i in range(batch_size)]

    target_attr = output[:batch_size,:,:20]
    # print(np.shape(target_attr))
    target = np.reshape(target_attr,(batch_size,n_objects,object_dim))
    target = target.float()

    # print(np.shape(data_time))
    # import pdb;pdb.set_trace()
    # data_time= np.squeeze(data_time)
    # output_index = [1+v for v in output_index]
    # print(data_time)
    # print(output_index)
    # print(data_time[output_index[0],:,:],data_time[0,:,:], ( data_time[output_index[0],:,:]-data_time[0,:,:]-MIN_T)/(MAX_T-MIN_T))
    duration_data = data_time #np.subtract(data_time[output_index,:,:] , data_time[:batch_size,:,:])

    duration_data_normalised = (duration_data - MIN_T)/(MAX_T-MIN_T)
    time_durations = np.repeat(duration_data_normalised, n_objects,axis=1)
    # print("time shape",np.shape(time_durations))
    # print(duration_data_normalised)
    # print(duration_data_normalised,data_time,data_time[output_index,:,:] , data_time[:batch_size,:,:],output_index,batch_size)
    # time_durations = np.expand_dims(time_durations,-1)
    time_durations=time_durations.float()

    objects = Variable(torch.FloatTensor(objects))
    time_durations       = Variable(torch.FloatTensor(time_durations))
    sender_relations   = Variable(torch.FloatTensor(sender_relations))
    receiver_relations = Variable(torch.FloatTensor(receiver_relations))
    rel_matrix      = Variable(torch.FloatTensor(rel_matrix))
    external_effect_info = Variable(torch.FloatTensor(external_effect_info))
    target             = Variable(torch.FloatTensor(target)).reshape(-1, object_dim)# 4 size output
                       
    if USE_CUDA:
        objects            = objects.cuda()
        time_durations       = time_durations.cuda()

        sender_relations   = sender_relations.cuda()
        receiver_relations = receiver_relations.cuda()
        rel_matrix      = rel_matrix.cuda()
        external_effect_info = external_effect_info.cuda()
        target             = target.cuda()
    
    return objects, time_durations, sender_relations, receiver_relations, rel_matrix, external_effect_info, target






def plot_search_results(grid):
    """
    Params: 
        grid: A trained GridSearchCV object.
    """
    #https://stackoverflow.com/a/57013458/8475746
    ## Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params=grid.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        #e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        #e_2 = np.array(stds_train[best_index])
        #ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        #ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
        ax[i].errorbar(x, y_1, linestyle='--', marker='o', label='test')
        ax[i].errorbar(x, y_2, linestyle='-', marker='^',label='train' )
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()