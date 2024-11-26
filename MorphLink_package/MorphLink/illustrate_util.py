import pandas as pd
import numpy as np
import cv2
import random 
import scanpy as sc
from scipy.sparse import issparse
import slideio
from .mask_util import  combine_masks


# most updated version (also incorporate with whole image illustration function)
def sample_illustration(f, adata, patch_info, patches, labels, plot_dir, num_cuts=5, range_step=1/4, num_sample=20, filter_mask_area=True, filter_cc_q100=False):
	channel=int(f.strip("log_")[1])
	f_type=f.strip("log_")[0]
	print("channel", channel, "f_type", f_type)
	#Create image for visual illustration
	patch_size=patches.shape[1]
	visual_img=np.ones([(patch_size+10)*num_sample-10,(patch_size*2+10+50*2)*num_cuts-50*2,3])*255
	#Select patches by quantile
	adata.obs[f]=adata.X[:, adata.var.index==f]
	m_area="m"+str(channel)+"_Area_of_1"
	if m_area in adata.var.index:
		adata.obs["mask_area"]=adata.X[:, adata.var.index==m_area].flatten().tolist()
	else:
		adata.obs["mask_area"]=adata.X[:, adata.var.index=="log_"+m_area].flatten().tolist()
	#Non mask filtered out
	adata=adata[adata.obs["mask_area"]>0]
	for i in range(num_cuts):
		mx=np.quantile(adata.obs[f], i/num_cuts+1/num_cuts*(1-range_step))
		mi=np.quantile(adata.obs[f], i/num_cuts+1/num_cuts*range_step)
		sub_tmp=adata[(adata.obs[f]>=mi)&(adata.obs[f]<=mx),:]
		if filter_mask_area:
			#========Filter on mask area========
			print(sub_tmp.obs["mask_area"])
			min_m_area=np.quantile(sub_tmp.obs["mask_area"], 0.25)
			max_m_area=np.quantile(sub_tmp.obs["mask_area"], 0.75)
			sub_tmp=sub_tmp[(sub_tmp.obs["mask_area"]>=min_m_area)&(sub_tmp.obs["mask_area"]<=max_m_area),:]
		if filter_cc_q100:
			#========Filter on cc area========
			cc_area_q100="c"+str(channel)+"_area_q100"
			if cc_area_q100 in sub_tmp.var.index:
				sub_tmp.obs["cc_area_q100"]=sub_tmp.X[:, sub_tmp.var.index==cc_area_q100].flatten().tolist()
			else:
				sub_tmp.obs["cc_area_q100"]=sub_tmp.X[:, sub_tmp.var.index=="log_"+cc_area_q100].flatten().tolist()
			min_cc_area_q100=np.quantile(sub_tmp.obs["cc_area_q100"], 0.2)
			max_cc_area_q100=np.quantile(sub_tmp.obs["cc_area_q100"], 0.8)
			sub_tmp=sub_tmp[(sub_tmp.obs["cc_area_q100"]>=min_cc_area_q100)&(sub_tmp.obs["cc_area_q100"]<=max_cc_area_q100),:]
		#===================================
		median_f=np.median(sub_tmp.obs[f])
		samples=sub_tmp.obs.index[(sub_tmp.obs[f]>=mi) & (sub_tmp.obs[f]<=mx)].tolist()[0:num_sample]
		print("Cut "+str(i)+", median="+str(median_f))
		print("Num of samples = ",len(samples))
		for j in range(len(samples)):
			sample=samples[j]
			id=patch_info.index.get_loc(sample)
			sample_patch=patches[id].copy()
			#Plot ori
			sample_label=labels[channel, id, ...]
			cv2.imwrite(plot_dir+'/cut_'+str(i)+"_sample_"+str(j)+'ori.png', sample_patch)
			#Plot mask
			if f_type=="m":
				alpha=1
				white_ratio=0
				color=[0, 255, 50]
				ret_img=patches[id].copy()
				sample_label=labels[channel, id, ...]
				ret_img=ret_img*(1-white_ratio)+np.array([255, 255, 255])*(white_ratio)
				ret_img[sample_label!=0]=ret_img[sample_label!=0]*(1-alpha)+np.array(color)*(alpha)
				ret_img=ret_img.astype(np.uint8)
				cv2.imwrite(plot_dir+'/cut_'+str(i)+"_sample_"+str(j)+'mask.png', ret_img)
			if f_type=="c":
				ret_img=np.zeros(patches[id].shape)
				for k in np.unique(sample_label):
					if k!=0:
						color=[random.randint(0,255),random.randint(0,255), random.randint(0,255)]
						ret_img[sample_label==k]=color
				ret_img=ret_img.astype(np.uint8)
				cv2.imwrite(plot_dir+'/cut_'+str(i)+"_sample_"+str(j)+'cc.png', ret_img)
			#Add sample patch
			visual_img[(patch_size+10)*j:(patch_size+10)*j+patch_size,(patch_size*2+10+50*2)*i:(patch_size*2+10+50*2)*i+patch_size,:]=sample_patch
			#Add image feature
			visual_img[(patch_size+10)*j:(patch_size+10)*j+patch_size,(patch_size*2+10+50*2)*i+patch_size+10:(patch_size*2+10+50*2)*i+patch_size*2+10,:]=ret_img
	#Create the whole image for illustration
	visual_img=visual_img.astype(np.uint8)
	cv2.imwrite(plot_dir+'/linkage_demonstration_'+f+'_ncuts='+str(num_cuts)+'_nsamples='+str(num_sample)+'.png', visual_img)
	return visual_img


#-------------------------------------------------------------------------------------
def mask_example(channel, adata, patch_info, patches, masks, plot_dir, num_samples=2, filter_mask_area=True):
	patch_size=patches.shape[1]
	ret_img=np.ones([(patch_size+5)*num_samples-5, patch_size*2+5,3])*255
	#========Filter on mask area========
	m_area="m"+str(channel)+"_Area_of_1"
	if m_area in adata.var.index:
		adata.obs["mask_area"]=adata.X[:, adata.var.index==m_area].flatten().tolist()
	else:
		adata.obs["mask_area"]=adata.X[:, adata.var.index=="log_"+m_area].flatten().tolist()
	sub_tmp=adata[adata.obs["mask_area"]>0]
	if sub_tmp.shape[0]<3:
		print("No example found for mask ", str(channel))
		return None
	else:
		min_m_area=np.quantile(sub_tmp.obs["mask_area"], 0.5)
		max_m_area=np.quantile(sub_tmp.obs["mask_area"], 0.8)
		sub_tmp=adata[(adata.obs["mask_area"]>=min_m_area)&(adata.obs["mask_area"]<=max_m_area),:]
		samples=random.sample(sub_tmp.obs.index.tolist(), num_samples) # randomness
	#===================================
	for j in range(len(samples)):
		sample=samples[j]
		id=patch_info.index.get_loc(sample)
		sample_patch=patches[id].copy()
		#Plot ori
		sample_mask=masks[channel, id, ...]
		tmp=sample_patch.copy()
		tmp[sample_mask!=0]=[0, 255, 50]
		ret_img[(patch_size+5)*j:(patch_size+5)*j+patch_size, 0:patch_size,:]=sample_patch
		ret_img[(patch_size+5)*j:(patch_size+5)*j+patch_size, patch_size+5:patch_size*2+5,:]=tmp
	ret_img=ret_img.astype(np.uint8)
	cv2.imwrite(plot_dir+'/sample_for_mask_'+str(channel)+'_.png', ret_img)
	return ret_img


#-------------------------------------------------------------------------------------
def mask_properity(masks, img, patch_info, d0, d1,center=True):
	per_contain=[]
	per_area=[]
	avg_rgb=[]
	combined_masks=combine_masks(masks, patch_info, d0, d1,center=True)
	for i in range(masks.shape[0]):
		ms=masks[i]
		areas=[]
		for j in range(ms.shape[0]):
			m=ms[j]
			areas.append(np.sum(m))
		non0_num=np.sum(np.array(areas)!=0)
		per_contain.append(np.round(non0_num/ms.shape[0], 3))
		per_area.append(np.round(np.sum(np.array(areas))/(non0_num*ms.shape[-1]*ms.shape[-2]), 3))
		#per_area.append(np.round(np.sum(np.array(areas))/(masks.shape[1]*masks.shape[2]*masks.shape[3]), 3))
		tmp=img[combined_masks[i]!=0]
		tmp=str(np.median(tmp, 0)[::-1])
		avg_rgb.append(tmp)
	ret=pd.DataFrame({"per_contain": per_contain, "per_area":per_area, "avg_rgb": avg_rgb})
	return ret

