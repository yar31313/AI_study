import SimpleITK as sitk
import numpy as np


def readTotalVolume(ct_volume_path): #데이터 불러오는 것
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(ct_volume_path)
    reader.SetFileNames(dicom_files)
    retrieved_ct_volume = reader.Execute()
    return retrieved_ct_volume

def resample_Total(input_volume, out_spacing=[1, 1, 1],  is_label=True):
    original_spacing = input_volume.GetSpacing()
    original_size = input_volume.GetSize()
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    #print(out_size)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(input_volume.GetDirection())
    resample.SetOutputOrigin(input_volume.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(input_volume.GetPixelIDValue())
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(input_volume)

def resize_Total(input_volume, width, height, depth):
    # Resize images to fixed spatial resolution in pixels
    # num_axial_slices = int(input_volume.GetSize()[-1])
    # output_size = [int(input_volume.GetSize()[0]), int(input_volume.GetSize()[1]), num_axial_slices]
    output_size = [width, height, depth]
    scale = np.divide(input_volume.GetSize(), output_size)
    spacing = np.multiply(input_volume.GetSpacing(), scale)
    transform = sitk.AffineTransform(3)
    resized_volume = sitk.Resample(input_volume, output_size, transform, sitk.sitkLinear, input_volume.GetOrigin(),
                                  spacing,input_volume.GetDirection())    
    return resized_volume

def extract_slices(image_volume):
    image_array = sitk.GetArrayFromImage(image_volume)
    image_slices_array = image_array[0::2, :, :]
    return image_slices_array

def normalize(volume):
    """Normalize the volume"""
    min = volume.min()
    max = volume.max()
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume