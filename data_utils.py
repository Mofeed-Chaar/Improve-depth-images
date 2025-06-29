import os

import glob
import config
import copy

import pandas as pd
class Datasets(object):
	def __init__(self):
		self.data = config.config('data.yaml')
		self.root = self.data.directories['root']

	def get_dirs(self):
		img_dir_left = os.path.join(self.root,self.data.directories['root_images'][0] )
		img_dir_right = os.path.join(self.root,self.data.directories['root_images'][1] )
		mask_dir = os.path.join(self.root,self.data.directories['root_labels'])

		img_train_left = glob.glob(os.path.join(img_dir_left,'train','*','*.png'))
		img_test_left = glob.glob(os.path.join(img_dir_left,'test','*','*.png'))
		img_val_left = glob.glob(os.path.join(img_dir_left,'val','*','*.png'))

		img_train_right = glob.glob(os.path.join(img_dir_right,'train','*','*.png'))
		img_test_right = glob.glob(os.path.join(img_dir_right,'test','*','*.png'))
		img_val_right = glob.glob(os.path.join(img_dir_right,'val','*','*.png'))

		mask_train = glob.glob(os.path.join(mask_dir,'train','*','*.png'))
		mask_test = glob.glob(os.path.join(mask_dir,'test','*','*.png'))
		mask_val = glob.glob(os.path.join(mask_dir,'val','*','*.png'))

		return {'img_train_left':img_train_left,'img_test_left':img_test_left,'img_val_left':img_val_left,
				'img_train_right':img_train_right,'img_test_right':img_test_right,'img_val_right':img_val_right,
				'mask_train':mask_train,'mask_test':mask_test,'mask_val':mask_val}

	def get_img_name(self,img_dir=None):
		image_name = '_'.join(img_dir.split('/')[-1].split('_')[:-1])
		return image_name
	def swap(self,list_name,first_elements,second_element):
		temp = list_name[first_elements]
		list_name[first_elements]= list_name[second_element]
		list_name[second_element] = temp
		return list_name 

	def search_in_list(self,list_dirs,str_name):
		for i in range(len(list_dirs)):
			if str_name in list_dirs[i]:
				return i
				break
		return None



	def sort_data_with_labels(self,img_dirs,mask_dirs):
		for i in range(len(img_dirs)):
			image_name = self.get_img_name(img_dirs[i])
			mask_name = self.get_img_name(mask_dirs[i])
			if image_name==mask_name:
				continue
			mask_correct_index = self.search_in_list(mask_dirs,image_name)
			mask_dirs = self.swap(mask_dirs,i,mask_correct_index)
		

			
		return img_dirs,mask_dirs

	def check_data_with_labels(self,img_dirs,mask_dirs):
		check = True
		for i in range(len(img_dirs)):
			image_name = self.get_img_name(img_dirs[i])
			mask_name = self.get_img_name(mask_dirs[i])
			if image_name==mask_name:
				continue
			print(image_name)
			print(f'here is false {i} the image name \n {img_dirs[i]} \n {mask_dirs[i]}')
			check = False
			break

		return check



if __name__ == '__main__':
	datasets = Datasets()
	data_dir = datasets.get_dirs()
	img_train_left = data_dir['img_train_left']
	img_train_right = data_dir['img_train_right']

	img_test_left = data_dir['img_test_left']
	img_test_right = data_dir['img_test_right']

	img_val_left = data_dir['img_val_left']
	img_val_right = data_dir['img_val_right']


	mask_train = data_dir['mask_train']
	mask_test = data_dir['mask_test']
	mask_val = data_dir['mask_val']


	print('sorting:')

	img_train_left,mask_train = datasets.sort_data_with_labels(img_train_left,mask_train)
	print('image train left sorted:')
	print(datasets.check_data_with_labels(img_train_left,mask_train))
	

	img_test_left,mask_test = datasets.sort_data_with_labels(img_test_left,mask_test)
	print('image test left sorted:')
	print(datasets.check_data_with_labels(img_test_left,mask_test))


	img_val_left,mask_val = datasets.sort_data_with_labels(img_val_left,mask_val)
	print('image val left sorted:')
	print(datasets.check_data_with_labels(img_val_left,mask_val))

#*******************************************************************************************************************************************	
	img_train_left,img_train_right = datasets.sort_data_with_labels(img_train_left,img_train_right)
	print('image train right sorted:')
	print(datasets.check_data_with_labels(img_train_right,mask_train))



	img_test_left,img_test_right = datasets.sort_data_with_labels(img_test_left,img_test_right)
	print('image train right sorted:')
	print(datasets.check_data_with_labels(img_test_right,mask_test))


	img_val_left,img_val_right = datasets.sort_data_with_labels(img_val_left,img_val_right)
	print('image train right sorted:')
	print(datasets.check_data_with_labels(img_val_right,mask_val))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	img_train_left +=img_test_left
	img_train_right +=img_test_right
	mask_train += mask_test
	data_train = {'img_train_left':img_train_left,'img_train_right':img_train_right,'mask_train':mask_train}
	#data_test = {'img_test_left':img_test_left,'img_test_right':img_test_right,'mask_test':mask_test}
	data_val = {'img_val_left':img_val_left,'img_val_right':img_val_right,'mask_val':mask_val}

	df_train = pd.DataFrame(data=data_train)
	#df_test = pd.DataFrame(data=data_test)
	df_val = pd.DataFrame(data=data_val)


	df_train.to_csv('data_trainE.csv', index=False)
	#df_test.to_csv('data_testE.csv', index=False)
	df_val.to_csv('data_valE.csv', index=False)



