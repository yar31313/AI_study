import SimpleITK as sitk
import numpy as np

def readTotalVolume(ct_volume_path):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(ct_volume_path)
    reader.SetFileNames(dicom_files)
    retrieved_ct_volume = reader.Execute()
    return retrieved_ct_volume

def resample_Total(input_volume, out_spacing=[1, 1, 1]):
    original_spacing = input_volume.GetSpacing()
    original_size = input_volume.GetSize()
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(input_volume.GetDirection())
    resample.SetOutputOrigin(input_volume.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(input_volume.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    
    return resample.Execute(input_volume)

def extract_slices(image_volume):
    image_array = sitk.GetArrayFromImage(image_volume)
    image_slices_array = image_array[:, :, :]
    return image_slices_array

def normalize(volume, thresholding=False):
    if thresholding:
        if type(thresholding)==int: thresholding=[thresholding, thresholding]
        upper_limit = np.percentile(volume, (1-thresholding[1])*100)
        volume[volume > upper_limit] = upper_limit
        lower_limit = np.percentile(volume, thresholding[0]*100)
        volume[volume < lower_limit] = lower_limit
        
    min = volume.min()
    max = volume.max()
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume