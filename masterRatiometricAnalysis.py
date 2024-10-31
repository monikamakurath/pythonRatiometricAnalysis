import firstStepLoadFile
import secondStepRegistration
import thirdStepSumChannels
import fourthStepSegmentation
import fifthStepRatioAnalysisPlots
import sixthStepRatioAnalysisImageStack

def main():
    print("Starting first step of ratiometric image analysis: Selecting the data file and exporting pre-registration.tiff stack")
    file_path, output_dir = firstStepLoadFile.firstStepLoadFile()

    if file_path and output_dir:
        print("Starting second step: Registering image stack using the loaded file...")
        secondStepRegistration.secondStepRegistration(file_path, output_dir)

        print("Starting third step: Summing intensities of registered image stack...")
        thirdStepSumChannels.thirdStepSumChannels(output_dir)

        print("Starting fourth step: Creating segmentation masks...")
        fourthStepSegmentation.fourthStepSegmentation(output_dir)

        print("Starting fifth step: Performing ratiometric analysis...")
        frame_rate = 9.65  # in seconds. Set the frame rate manually here or prompt the user if needed
        fifthStepRatioAnalysisPlots.fifthStepRatioAnalysisPlots(output_dir, frame_rate)

        print("Starting sixth step: Generating ratiometric images with annotations...")
        pixel_width = 0.0425232
        scale_bar_length_microns = 1
        high_dpi = 300
        sixthStepRatioAnalysisImageStack.sixthStepRatioAnalysisImageStack(output_dir, frame_rate, pixel_width, scale_bar_length_microns, high_dpi)

    else:
        print("Error: File path or output directory not provided.")

if __name__ == "__main__":
    main()








