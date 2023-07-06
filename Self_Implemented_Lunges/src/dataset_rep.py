from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
import numpy as np
import pandas as pd
import params
from utils_rep import normalize_time
# pd.options.mode.chained_assignment='raise'

# warnings.simplefilter(action='always', category=pd.errors.PerformanceWarning)
import statsmodels.api as sm

class LandmarksDataset(Dataset):

    def __init__(self, xl_file, root_dir, future_offset=1,fixed_offset=1, transform=None):
        """
        Args:A
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        df_map = self.process_data_from_excel(xl_file)
        
#         df1.
# df2.reset_index(drop=True, inplace=True)
        # import pdb;pdb.set_trace()
        self.MIN_T,self.MAX_T = normalize_time(df_map[list(df_map.keys())[0]].iloc[:,[0]].values, future_offset)# choose any ,they should be similar????????????
        # reserved_df_map={}
        # for key in df_map.keys():
        #     reserved_df_map[key] = df_map[key].iloc[-100:,:]
        #     df = df.iloc[:-100,:]
        #     df = pd.concat([df.reset_index(drop=True),reserved_df.reset_index(drop=True)],axis=0)
        # reserved_df_map = df.iloc[-100:,:]




        # df=df.astype(float)
        self.landmarks_frame_map =df_map
        # print("size",df.shape)
        # print(self.landmarks_frame.head(3))
        self.root_dir = root_dir
        self.transform = transform
        self.future_offset=future_offset
        self.test=False

        # self.future_offset = future_offset
        # df['time'] = df.iloc[out_index,[0]].reset_index(drop=True) -df.iloc[:,[0]].reset_index(drop=True)

    def max_min(self,df,select):
        mn = 10.0
        mx = -10.0

        for i in select:
            mn = min(mn, min(df[i]))
            mx = max(mx, max(df[i]))
        return mx,mn
    def smooth(self,df,select,norm = True,f = 0):        
        new_df = pd.DataFrame()
        if f == 0:
            f = min( 0.03, (41 / len(df)) )
        time = [i for i in range(len(df))]
        
        all = select
        '''for i in range(33):
            all.append('x_'+str(i))
            all.append('y_'+str(i))
            all.append('z_'+str(i))
        print(all)'''

        for i in all:
            new_df[i] = sm.nonparametric.lowess(df[i].values, time,frac= f,
                                                    it=3, delta=0.0, is_sorted=True,
                                                    missing='drop', return_sorted=False)
        
        if norm:
            x_max, x_min = self.max_min(new_df,all[1::3])
            y_max, y_min = self.max_min(new_df,all[2::3])
            z_max, z_min = self.max_min(new_df,all[3::3])

            #print("x = {} : {}\ny = {} : {}\nz = {} : {}".format(x_min,x_max,y_min,y_max,z_min,z_max))

            # normalise x
            for i in select[1::3]:
                new_df[i] = (new_df[i] - x_min) / (x_max - x_min)

            #normalise y
            for i in select[2::3]:
                new_df[i] = 1 - ( (new_df[i] - y_min) / (y_max - y_min) )

            #normalise z
            for i in select[3::3]:
                new_df[i] = (new_df[i] - z_min) / (z_max - z_min)
            
        '''x_max, x_min = max_min(new_df,select[0::3])
        y_max, y_min = max_min(new_df,select[1::3])
        z_max, z_min = max_min(new_df,select[2::3])

        print("x = {} : {}\ny = {} : {}\nz = {} : {}".format(x_min,x_max,y_min,y_max,z_min,z_max))'''
        
        new_df = new_df[select]
            
        return new_df

    def shift_cols(self,df):
        right_knee='26'
        left_knee = '25'
        right_heel = '30'
        left_heel = '29'
        right_toe = '32'
        left_toe = '31'

        if sum(df.iloc[:10]['x'+left_knee]) < sum(df.iloc[:10]['x'+right_knee]): # if true right knee is ahead of left knee
            front_knee = right_knee
            back_knee = left_knee
            front_heel = right_heel
            back_toe = left_toe
        else:
            # import pdb;pdb.set_trace()
            front_knee=left_knee
            back_knee=right_knee
            front_heel = left_heel
            back_toe = right_toe
#  all our data used is right facing
            
        # if  df['x'+right_knee].isin([-1]).any().any():
        #     right_shoulder = left_shoulder
        #     right_hip = left_hip
        #     right_knee = left_knee
        if df['x'+right_knee].isin([-1]).any().any() or df['x'+left_knee].isin([-1]).any().any():
            print("BOTH LEFT AND RIGHT knee has missing data")



        df_org=df.copy()
        coords_id = params.coords_ids # right shulder, right hip, font knee,back knee, front heel, back toe
        target_id = [front_knee,back_knee,front_heel,back_toe]
        for id,val in enumerate(coords_id[2:6]):
            mask = ['x'+str(val), 'y' + str(val), 'z' + str(val) ]
            target = ['x'+str(target_id[id]), 'y' + str(target_id[id]), 'z' + str(target_id[id]) ]
            df_org[mask ] = df[target]
        return df_org

    def pre_process(self,df):
        cols = [2]
        for i in range(132):
            if i%4:
                cols.append(i+5)
        # print("cols here",df.columns)
        # print(cols)
        df = df[cols]
        labels = ['time']
        cord = {}
        cord[0] = 'x'
        cord[1] = 'y'
        cord[2] = 'z'
        for i in range(99):
            t = i // 3
            c = i % 3
            labels.append(cord[c]+str(t))

        df.columns = labels # x0,y0,z0,x1,y1,z1,x2.....
#         select = ['time','x0','y0','z0','x23','y23','z23','x24','y24','z24','x25','y25','z25','x26','y26','z26']
        # select = ['time','x0','y0', 'x24','y24','x23','y23','x26','y26','x25','y25'] #SQUATS
        # print(df.columns)

        
        df=self.shift_cols(df)
        # mask = ['x'+str(coords_id[]), 'y' + str(coords_id[2]), 'z' + coords_id[2] ]
        # target = ['x'+str(front_knee), 'y' + str(front_knee), 'z' + str(front_knee) ]
        # df_org.loc[:,mask ] = df[:,target]

        coords_id = params.coords_ids
        # print("top botoom",df.columns)
        select = ['time']
        for coords in coords_id:
            select.append('x'+str(coords))
            select.append('y'+str(coords))
            select.append('z'+str(coords))

        # select = ['time','x12','y12', 'x14','y14','x16','y16','x24','y24','x26','y26','x28','y28'] #shoulder press

        # print("SIZE = %d before 0 removal"%(len(df)))
        if len(df) == 0:
            return df
        if df[select[1:-6:3]].isin([-1]).any().any():
            # import pdb;pdb.set_trace()
            print(df[select[1:-6:3]].isin([-1]).any())
            return pd.DataFrame()
        # print("SIZE = %d after 0 removal\n\n"%(len(df)))
        # import pdb;pdb.set_trace()
# THIS DID THE TRICK
        df = self.add_staionary_points(df)
        df = self.smooth(df,select)
 
    
#         velocity
        for i in range(1,len(df.columns),3):
            df.loc[:,'d'+df.columns[i]] = (df[df.columns[i]]-df[df.columns[i]].shift(1)).fillna(0) # put displacement/velocity(dx)
            df.loc[:,'d'+df.columns[i+1]] = (df[df.columns[i+1]]-df[df.columns[i+1]].shift(1)).fillna(0) # put displacement/velocity(dy)
            df.loc[:,'d'+df.columns[i+2]] = (df[df.columns[i+1]]-df[df.columns[i+1]].shift(1)).fillna(0) # put displacement/velocity(dz)

        # print(df.columns)
    #     select = ['time','x0','y0','dx0', 'dy0', 'x24','y24','dx24', 'dy24', 'x23','y23','dx23', 'dy23','x26','y26',  
    #    'dx26', 'dy26','x25','y25','dx25', 'dy25']
        select = ['time']
        for coords in coords_id:
            select.append('x'+str(coords))
            select.append('y'+str(coords))
            select.append('dx'+str(coords))
            select.append('dy'+str(coords))

        df = df[select]


 

#     JOINTS
        # import pdb;pdb.set_trace()
        org_cols = len(df.columns)
        df_copy= df.copy()
        cols_dict={}
        for i in range(1,org_cols,4):
            for j in range(1,org_cols,4):
                if i!=j:
                    # if str(df.columns[i]) == 'x34':
                    #     import pdb;pdb.set_trace()
                    # print('@'+str(df.columns[i])+str(df.columns[j]) , (df_copy.iloc[:,i] - df_copy.iloc[:,j]))
                    # df.loc[:,'r'+str(df.columns[i])+str(df.columns[j])] =( (df_copy.iloc[:,i] - df_copy.iloc[:,j])**2 + (df_copy.iloc[:,i+1] - df_copy.iloc[:,j+1])**2 )**(1/2) # distance b/w joints
                    # df.loc[:,'@'+str(df.columns[i])+str(df.columns[j])] =np.arctan( (df_copy.iloc[:,i+1] - df_copy.iloc[:,j+1]) /(df_copy.iloc[:,i] - df_copy.iloc[:,j])) # angle b/w joints
                    cols_dict['r'+str(df.columns[i])+str(df.columns[j])] = ( (df.iloc[:,i] - df.iloc[:,j])**2 + (df.iloc[:,i+1] - df.iloc[:,j+1])**2 )**(1/2) 
                    cols_dict['@'+str(df.columns[i])+str(df.columns[j])] = np.arctan( (df.iloc[:,i+1] - df.iloc[:,j+1]) /(df.iloc[:,i] - df.iloc[:,j])) 
# IMPORTANT   
         

        new_df = pd.DataFrame(cols_dict) 
        df_copy = pd.concat([df_copy, new_df], axis=1)  

        df_copy = df_copy.iloc[1:,:] # REMOVE FIRST VALUES AS VELOCITY DATA IS ERRONIOUS HERE 0 for both x and y
        return df_copy 


    def process_data_from_excel(self,file_path='data/Squats_correct.xlsx'):
        sheet_to_df_map = pd.read_excel(file_path, sheet_name=None)
        arr=[]
        data={}
        data_time={}
        sheet_to_df_map_processed = {}  # this is an additional check to remove bad values but it removes many good values also
        new_key = 0
        for key in sheet_to_df_map.keys():

            df = sheet_to_df_map[key] 
            # print(df.columns,"process_data fotm excel")
            if "Unnamed: 0" in df.columns:
                df = df.drop("Unnamed: 0", axis = 1)
            # df['time'] -= df.iloc[0,0]
            df = self.pre_process( df)
            if len(df) < 20: # sometimes very few datapoints remain or even 0 remain while removing - what to od?
                print(file_path,"df key org rejected",key,len(df))

                continue
            else:
                print(file_path,"df key org accepted",key,len(df))
            # if len(df) > 8:
            sheet_to_df_map_processed[str(new_key)] = df
                # print("df key",str(new_key),len(sheet_to_df_map_processed[str(new_key)] ))
            new_key+=1

        # print("wrote excel file")
        # writer = pd.ExcelWriter('processed.xlsx', engine='xlsxwriter')
        # for key in sheet_to_df_map_processed.keys():
            
        #     temp =  sheet_to_df_map_processed[key]
        #     temp.to_excel(writer,sheet_name=str(key))
        # writer.save()

        return sheet_to_df_map_processed
    def add_staionary_points(self,df):
        assert len(df.columns) == 100, 'Mediapipe not used for pose estimation '+str(len(df.columns))
        # print(df.head())
        top_id = (len(df.columns)-1)//3
        bottom_id = top_id +1
        col_index = np.repeat([top_id,bottom_id],3)
        vals= [.5,0,0,.5,1,0]
        for i,j,k in zip(['x','y','z','x','y','z'],col_index,vals):
            df.loc[:,str(i)+str(j)] = k
        
        return df


    def __len__(self,):
        return len(self.landmarks_frame_map)


    def __getitem__(self, idx, ):
        # randomly choose any of the rep as train/test data
        # df_key = np.random.choice(list( self.landmarks_frame_map.keys()))
        # print("selected df",idx)
        # print(self.landmarks_frame_map.keys())
        self.landmarks_frame = self.landmarks_frame_map[str(idx.tolist()[0])]
        batch_size = len(self.landmarks_frame['time'])
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # idx= random.sample(range(0, 32), 26)
        idx = np.arange(0,batch_size)
        # print("IDX BEFORE",idx)

        idx = [i for i in idx if i < len(self.landmarks_frame)-1]
        # print("IDX after",idx)
        out_idx = [1+np.random.randint(i,min(i+self.future_offset,len(idx)) ) for i in range(len(idx))]
        if self.test: # for testing case
            print(idx,"testing",out_idx)
            self.future_offset=1
            out_idx = [min(i+self.future_offset,len(idx)) for i in range(len(idx))]
        # out_idx = [i+1 for i in idx] # THIS COULD CREATE OFF by 1 errro should be handled while enumerating dataloader,will above suffice?
        # import pdb;pdb.set_trace()
        # print(idx,out_idx)
        time = self.landmarks_frame.iloc[out_idx,[0]].reset_index(drop=True) - self.landmarks_frame.iloc[idx, [0]].reset_index(drop=True) 
        time = np.array(time)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array(landmarks)
        # print("shape",np.shape(landmarks))
#         print(landmarks[0])
        output_landmarks = self.landmarks_frame.iloc[out_idx, 1:]
        output_landmarks = np.array(output_landmarks)
        # print("shape",np.shape(output_landmarks))
#         landmarks = landmarks.astype('float').reshape(1, 2)
        sample = {"time": time, 'landmarks': landmarks, "output":output_landmarks, "col_id":self.landmarks_frame.columns[1:]}

        if self.transform:
            sample = self.transform(sample)
        return sample
