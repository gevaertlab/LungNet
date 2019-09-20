from scipy.ndimage import zoom
import imageio as imio
from matplotlib import path
import os
import pydicom as dicom
import numpy as np

target_size = 32
root_dir = './Varian2/'


def get_transform_matrix(sorted_images):
    N = len(sorted_images)
    if N<2:
        return None
    T1 = np.array(sorted_images[0].ImagePositionPatient).astype(np.float32)
    TN = np.array(sorted_images[len(sorted_images)-1].ImagePositionPatient).astype(np.float32)
    cosines = sorted_images[0].ImageOrientationPatient
    delta_r = sorted_images[0].PixelSpacing[0]
    delta_c = sorted_images[0].PixelSpacing[1]
    F = np.array([[cosines[3],cosines[0]],[cosines[4],cosines[1]],[cosines[5],cosines[2]]])
    transform = np.zeros([4,4])
    transform[0:3,0]= F[:,0]*delta_r
    transform[0:3,1] = F[:,1]*delta_c
    transform[0:3,2] = ((T1-TN)/(1-N)).T
    transform[0:3,3] = T1.T
    transform[3,3]=1
    return transform

def load_series(location):
    # loads the dicom series and returns the sorted list of dicom image files, the rtstruct file and the average slice spacing
    files = [dicom.read_file(os.path.join(location,f)) for f in os.listdir(location) if f.endswith('dcm')]
    ct_files = [f for f in files if f.SOPClassUID =='1.2.840.10008.5.1.4.1.1.2']
    rt_struct_file = [f for f in files if f.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3' ]
    if len(rt_struct_file)>1:
        print('Found more than one structure file in the directory, will pick the first one')        
    elif len(rt_struct_file)==0:
        print('Found no structure file in the directory')
        return None, None, None
    rt_struct_file = rt_struct_file[0]
    files = sorted(ct_files,key=lambda x:x.SliceLocation) # basic sort (may not be the correct ordering)
    # find actual ordering
    orientation = np.array(files[0].ImageOrientationPatient)
    vector_perp = np.cross(orientation[0:3],orientation[3:])
    unit_vector_perp = vector_perp/(np.dot(vector_perp,vector_perp))**0.5
    files2 = sorted(files,key=lambda x:np.dot(np.array(x.ImagePositionPatient),unit_vector_perp)) # this should be the correct ordering
    # find slice spacing
    slice_spacing1 = np.dot((np.array(files2[1].ImagePositionPatient) - np.array(files2[0].ImagePositionPatient)),unit_vector_perp)
    avg_spacing = np.dot((np.array(files2[-1].ImagePositionPatient) - np.array(files2[0].ImagePositionPatient)),unit_vector_perp)/(len(files2)-1) 
    if abs(slice_spacing1-avg_spacing)>avg_spacing/(len(files2)-1):
        print('maybe missing slices')

    return files2,rt_struct_file,avg_spacing

def load_pixel_vol(vol,axis):
    # loads the pixel array of the dicom volume and concatenates the slices along specified axis
    return np.stack([image.pixel_array for image in vol],axis=axis)

def transform_coords(t_matrix,coords):
    # utility to transfrom coords given a transform matrix
    concat_coord = np.concatenate((coords,np.ones([coords.shape[0],1])),axis =1)
    new_coords = np.dot(t_matrix,concat_coord.T)
    return new_coords[0:3,:]

def get_roi_info(mask_info):
    # parses an rtstruct file and returns a dictionary with the contour information
 #   mask_info = dicom.read_file(mask_info)
    contour_dict = {}
    gtv_numbers = []
    for s in mask_info.StructureSetROISequence:
        if 'GTV' in s.ROIName.upper():
            gtv_numbers.append(s.ROINumber)
    for i in range(len(mask_info.ROIContourSequence)):
        if mask_info.ROIContourSequence[i].ReferencedROINumber not in gtv_numbers:
            continue
        contour_dict['ROIContourSequence'+str(i)] = {}
        for j in range(len(mask_info.ROIContourSequence[i].ContourSequence)):
            contour_data = mask_info.ROIContourSequence[i].ContourSequence[j].ContourData
            coords = [[contour_data[3*i],contour_data[3*i+1],contour_data[3*i+2]] for i in range(int(len(contour_data)/3))] 
            sop_uid = mask_info.ROIContourSequence[i].ContourSequence[j].ContourImageSequence[0].ReferencedSOPInstanceUID
            contour_dict['ROIContourSequence'+str(i)][sop_uid] = coords
    return contour_dict

def process_contours(contour_dict,ref_vol):
    # creates masks from each contour
    voxel_2_world = get_transform_matrix(ref_vol)
    x_len = ref_vol[0].pixel_array.shape[0]
    y_len = ref_vol[0].pixel_array.shape[1]
    z_len = len(ref_vol)
    slice_grid = np.array(np.where(np.zeros([x_len,y_len])==0)).T
    contour_points = []
    masks = []
    for roi_contour_seq in contour_dict:
        new_vol = np.zeros([ref_vol[0].pixel_array.shape[0],ref_vol[0].pixel_array.shape[1],len(ref_vol)])
        for sop_id in contour_dict[roi_contour_seq]:
            coords = np.array(contour_dict[roi_contour_seq][sop_id])
            roi_t2_pixel_coords = transform_coords(np.linalg.inv(voxel_2_world),coords)
            slice_to_process = np.unique(np.round(roi_t2_pixel_coords[2:]).astype(np.int32))
            if slice_to_process.shape[0]>1:
                print('Slices to process:', slice_to_process)
                return None
            contour_path = path.Path(roi_t2_pixel_coords[0:2,:].T,closed=True)
            contour_points.append(roi_t2_pixel_coords)
            new_slice = np.zeros([x_len,y_len,1])
            points_included = slice_grid[np.where(contour_path.contains_points(slice_grid))]
            new_slice[points_included[:,0],points_included[:,1],np.zeros_like(points_included[:,1])]=1
            new_vol[:,:,slice_to_process]=new_slice
        masks.append(new_vol)
    return masks,contour_points

def extract_roi(vol,cube_centers,pixel_cube_dims):
    # extracts the roi with zero-padding if necessary
    xlen, ylen, zlen = np.ceil(0.5*pixel_cube_dims).astype(np.int16)
    padding = tuple([(xlen,xlen),(ylen,ylen),(zlen,zlen)])
    padded_vol = np.pad(vol,pad_width=padding,mode='constant')
    new_centers = np.round(np.array([xlen,ylen,zlen])+cube_centers).astype(np.int16)
    return padded_vol[int(new_centers[0])-xlen:int(new_centers[0])+xlen, new_centers[1]-ylen:new_centers[1]+ylen, new_centers[2]-zlen:new_centers[2]+zlen]

def resize_tumor(vol,mask,target_size,spacings):
    # resizes the roi to the fixed target_size 
    non_zero_locations = np.where(mask!=0)
    x_min, x_max = non_zero_locations[0].min(), non_zero_locations[0].max()
    y_min, y_max = non_zero_locations[1].min(), non_zero_locations[1].max()
    z_min, z_max = non_zero_locations[2].min(), non_zero_locations[2].max()
    cube_centers = 0.5*(np.array([x_max,y_max,z_max])+np.array([x_min,y_min,z_min]))
    dims_world = np.multiply((np.array([x_max,y_max,z_max])-np.array([x_min,y_min,z_min])),spacings)
    max_dimension = dims_world.max()      
    real_cube_dims = np.array([max_dimension]*3)
    pixel_cube_dims = np.round(np.divide(real_cube_dims,spacings))
    cropped_vol = extract_roi(vol,cube_centers,pixel_cube_dims)
    cropped_mask = extract_roi(mask,cube_centers,pixel_cube_dims)
    resized_vol = zoom(cropped_vol,zoom=target_size/np.array(cropped_vol.shape),prefilter=False)
    resized_mask = zoom(cropped_mask,zoom = target_size/np.array(cropped_mask.shape),prefilter=False)
    return np.stack([resized_vol,resized_mask],axis=3)

def resize_tumor2(vol, mask,  target_size, spacings,intermediate_common_size = np.array([128,128,49])):
    # this is the method used to resize by Mu and Ed: for each tumor, find its bounding box, for each slice, zoom the bounding box to 
    # 128 x 128 (do not zoom along z-axis), along the z-axis, extract only upto the bounding box edge, zero-pad to 49 slices
    # finally, resize the 128 x 128 x 49 volume to 32 x 32 x 32
    non_zero_locations = np.where(mask!=0)
    x_min, x_max = non_zero_locations[0].min(), non_zero_locations[0].max()
    y_min, y_max = non_zero_locations[1].min(), non_zero_locations[1].max()
    z_min, z_max = non_zero_locations[2].min(), non_zero_locations[2].max()
    pixel_dims = np.array([x_max,y_max,z_max])-np.array([x_min,y_min,z_min])
    int_volume = vol[x_min:x_max,y_min:y_max,z_min:z_max]
    int_target_size = np.array([intermediate_common_size[0],intermediate_common_size[1],pixel_dims[2]])
    int_volume = zoom(int_volume,zoom = np.divide(int_target_size,pixel_dims),prefilter=False)
    padding = tuple([(0,0),(0,0),(0,intermediate_common_size[2]-min(intermediate_common_size[2],pixel_dims[2]))])
    padded_vol = np.pad(int_volume, pad_width=padding,mode='constant')
    cropped_vol = padded_vol[:,:,0:intermediate_common_size[2]]
    resized_vol = zoom(cropped_vol,zoom = target_size/intermediate_common_size,prefilter=False)
    
    int_mask = mask[x_min:x_max,y_min:y_max,z_min:z_max]
    int_mask = zoom(int_mask,zoom = np.divide(int_target_size,pixel_dims),mode='constant',prefilter=False)
    padded_mask = np.pad(int_mask, pad_width=padding,mode='constant')
    cropped_mask = padded_mask[:,:,0:intermediate_common_size[2]]
    resized_mask = zoom(cropped_mask,zoom = target_size/intermediate_common_size,mode='constant',prefilter=False)
    resized_mask[np.where(resized_mask>0)]=1
    return np.stack([resized_vol,resized_mask],axis=3)

def normalize(vol):
    return (vol - vol.min())/(vol.max()-vol.min())

def process_directory(vol_path,save_gifs=True,save_dir='./'):
    vol,rt_struct_file,slice_spacing = load_series(vol_path)
    if vol is None:
        return []
    roi_info = get_roi_info(rt_struct_file)
    masks,_ = process_contours(roi_info,vol)
    pixel_vol = load_pixel_vol(vol,axis=2)
    spacings = np.array([vol[0].PixelSpacing[0],vol[0].PixelSpacing[1],slice_spacing])
    count = 0
    input_tumors = []
    for mask in masks:
        count+=1
        input_tumor = resize_tumor2(pixel_vol,mask,target_size,spacings)
        input_tumors.append(input_tumor)
        if save_gifs:
            normal_tumor = normalize(input_tumor[:,:,:,0])
            imio.mimwrite(os.path.join(save_dir,str(count)+'.gif'),[np.concatenate([normal_tumor[:,:,i],np.ones([32,10]),input_tumor[:,:,i,1]],axis=1) for i in range(target_size)])
        
    return input_tumors
if __name__ == '__main__':
    process_directory(root_dir)