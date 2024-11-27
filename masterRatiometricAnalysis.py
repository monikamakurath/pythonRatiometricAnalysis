import firstStepLoadFile
import secondStepRegistration
import thirdStepSumChannels
import fourthStepSegmentation
import fifthStepRatioAnalysisPlots
import sixthStepRatioAnalysisImageStack

def main():
    print("First: Selecting the data file and exporting pre-registration.tiff stack")
    file_path, output_dir = firstStepLoadFile.firstStepLoadFile()

    if file_path and output_dir:
        print("1: Registering image stack using the loaded file...")
        secondStepRegistration.secondStepRegistration(file_path, output_dir)

        print("2: Summing intensities of registered image stack...")
        thirdStepSumChannels.thirdStepSumChannels(output_dir)

        print("3: Creating segmentation masks...")
        fourthStepSegmentation.fourthStepSegmentation(output_dir)

        print("4: Performing ratiometric analysis...")
        frame_rate = 59.86  # in seconds. Set the frame rate manually here or prompt the user if needed
        fifthStepRatioAnalysisPlots.fifthStepRatioAnalysisPlots(output_dir, frame_rate)

        print("5: Generating ratiometric images with annotations...")
        pixel_width = 0.0425270
        scale_bar_length_microns = 10
        high_dpi = 300
        sixthStepRatioAnalysisImageStack.sixthStepRatioAnalysisImageStack(output_dir, frame_rate, pixel_width, scale_bar_length_microns, high_dpi)

    else:
        print("Error: File path or output directory not provided.")

if __name__ == "__main__":
    main()








