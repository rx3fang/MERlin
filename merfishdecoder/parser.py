import sys
import argparse
import merfishdecoder

def parse_args():
    parser = argparse.ArgumentParser(
        description = "You are using merfishdecoder")
    subparsers = parser.add_subparsers(
        title="functions",
        dest="command",metavar="")
    
    add_create_analysis(subparsers)
    add_registration(subparsers)
    add_preprocessing(subparsers)
    add_predict_prob(subparsers)
    add_decoding(subparsers)
    add_train_psm(subparsers)
    add_extract_barcodes(subparsers)
    add_extract_pixel_traces(subparsers)
    add_export_barcodes(subparsers)
    add_filter_barcodes(subparsers)
    add_segmentation(subparsers)
    add_extract_features(subparsers)
    add_export_features(subparsers)
    add_filter_features(subparsers)
    add_barcode_assignment(subparsers)
    add_export_gene_feature_matrix(subparsers)
    add_estimate_bit_error(subparsers)

    if len(sys.argv) > 1:
        if (sys.argv[1] == '--version' or sys.argv[1] == '-v'):
            print(merfishdecoder.__version__)
            exit()
        args = parser.parse_args()
    else:
        args = parser.parse_args(["-h"])
        exit()

    if args.command == "create-analysis":
        from merfishdecoder.apps import run_create_analysis
        run_create_analysis.run_job(
            dataSetName=args.data_set_name,
            codebookName=args.codebook_name,
            dataOrganizationName=args.data_organization_name,
            microscopeParameterName=args.microscope_parameter_name,
            positionName=args.position_name,
            microscopeChromaticAberrationName=args.microscope_chromatic_aberration_name,
            dataHome=args.data_home,
            analysisHome=args.analysis_home)
    elif args.command == "register-images":
        from merfishdecoder.apps import run_registration
        run_registration.run_job(
            dataSetName=args.data_set_name,
            fov=args.fov,
            zpos=args.zpos,
            outputName=args.output_name,
            refFrameIndex=args.ref_frame_index,
            registerDrift=args.register_drift,
            highPassFilterSigma=args.high_pass_filter_sigma,
            registerColor=args.register_color,
            registerColorProfile=args.register_color_profile,
            saveFiducials=args.save_fiducials)               
    elif args.command == "process-images":
        from merfishdecoder.apps import run_preprocessing
        run_preprocessing.run_job(
            dataSetName=args.data_set_name,
            fov=args.fov,
            zpos=args.zpos,
            warpedImagesName=args.warped_images_name,
            outputName=args.output_name,
            highPassFilterSigma=args.high_pass_filter_sigma,
<<<<<<< HEAD
            lowPassFilterSigma=args.low_pass_filter_sigma)                             
=======
            scaleFactorFile=args.scale_factor_file,
            logTransform=args.log_transform)
    elif args.command == "predict-prob":
        from merfishdecoder.apps import run_predict_prob
        run_predict_prob.run_job(
            dataSetName=args.data_set_name,
            fov=args.fov,
            zpos=args.zpos,
            processedImagesName=args.processed_images_name,
            outputName=args.output_name,
            modelName=args.model_name,
            kernelSize=args.kernel_size)
>>>>>>> c1e3ee130d7256ac122fa56c033538dda702739b
    elif args.command == "decode-images":
        from merfishdecoder.apps import run_decoding
        run_decoding.run_job(
            dataSetName=args.data_set_name,
            fov=args.fov,
            zpos=args.zpos,
            decodingImagesName=args.decoding_images_name,
            outputName=args.output_name,
            maxCores=args.max_cores,
            borderSize=args.border_size, 
            magnitudeThreshold=args.magnitude_threshold,
            distanceThreshold=args.distance_threshold)
    elif args.command == "train-psm":
        from merfishdecoder.apps import run_train_psm
        run_train_psm.run_job(
            dataSetName=args.data_set_name,
            decodedImagesDir=args.decoded_images_dir,
            outputName=args.output_name,
            zposNum=args.zpos_num)
    elif args.command == "extract-barcodes":
        from merfishdecoder.apps import run_extract_barcodes
        run_extract_barcodes.run_job(
            dataSetName=args.data_set_name,
            fov=args.fov,
            zpos=args.zpos,
            decodedImagesName=args.decoded_images_name,
            outputName=args.output_name,
            psmName=args.psm_name,
            barcodesPerCore=args.barcodes_per_core,
            maxCores=args.max_cores)
    elif args.command == "extract-pixel-traces":
        from merfishdecoder.apps import run_extract_pixel_traces
        run_extract_pixel_traces.run_job(
            dataSetName=args.data_set_name,
            fov=args.fov,
            zpos=args.zpos,
            areaSizeThreshold=args.area_size_threshold,
            magnitudeThreshold = args.magnitude_threshold,
            distanceThreshold=args.distance_threshold,
            extractedBarcodesName=args.extracted_barcodes_name,
            processedImagesName=args.processed_images_name,
            decodedImagesName=args.decoded_images_name,
            outputName=args.output_name)
    elif args.command == "export-barcodes":
        from merfishdecoder.apps import run_export_barcodes
        run_export_barcodes.run_job(
            dataSetName=args.data_set_name,
            decodedBarcodesDir=args.decoded_barcodes_dir,
            outputName=args.output_name)
    elif args.command == "filter-barcodes":
        from merfishdecoder.apps import run_filter_barcodes
        run_filter_barcodes.run_job(
            dataSetName=args.data_set_name,
            exportedBarcodesName=args.exported_barcodes_name,
            outputName=args.output_name,
            fovNum=args.fov_num,
            keepBlankBarcodes=args.keep_blank_barcodes,
            misIdentificationRate=args.mis_identification_rate,
            minAreaSize=args.min_area_size)
    elif args.command == "segmentation":
        from merfishdecoder.apps import run_segmentation
        run_segmentation.run_job(
            dataSetName=args.data_set_name,
            fov=args.fov,
            zpos=args.zpos,
            warpedImagesName=args.warped_images_name,
            outputName=args.output_name,
            featureName=args.feature_name,
            diameter=args.diameter,
            modelType=args.model_type,
            gpu=args.gpu)
    elif args.command == "extract-features":
        from merfishdecoder.apps import run_extract_features
        run_extract_features.run_job(
            dataSetName=args.data_set_name,
            fov=args.fov,
            zpos=args.zpos,
            segmentedImagesName=args.segmented_images_name,
            outputName=args.output_name)
    elif args.command == "export-features":
        from merfishdecoder.apps import run_export_features
        run_export_features.run_job(
            dataSetName=args.data_set_name,
            segmentedFeaturesDir=args.segmented_features_dir,
            outputName=args.output_name,
            bufferSize=args.buffer_size)
    elif args.command == "filter-features":
        from merfishdecoder.apps import run_filter_features
        run_filter_features.run_job(
            dataSetName=args.data_set_name,
            exportedFeaturesName=args.exported_features_name,
            outputName=args.output_name,
            minZplane=args.min_zplane,
            borderSize=args.border_size)
    elif args.command == "assign-barcodes":
        from merfishdecoder.apps import run_barcode_assignment
        run_barcode_assignment.run_job(
            dataSetName=args.data_set_name,
            exportedBarcodesName=args.exported_barcodes_name,
            exportedFeaturesName=args.exported_features_name,
            outputName=args.output_name,
            maxCores=args.max_cores,
            bufferSize=args.buffer_size)
    elif args.command == "export-gene-feature-matrix":
        from merfishdecoder.apps import run_export_gene_feature_matrix
        run_export_gene_feature_matrix.run_job(
            dataSetName=args.data_set_name,
            barcodesName=args.barcodes_name,
            featuresName=args.features_name,
            outputName=args.output_name,
            maxCores=args.max_cores)
    elif args.command == "estimate-bit-error":
        from merfishdecoder.apps import run_estimate_bit_err
        run_estimate_bit_err.run_job(
            dataSetName=args.data_set_name,
            pixelTracesDir=args.pixel_traces_dir,
            outputName=args.output_name)
    else:
        print("command %s not found" % args.command)

def add_create_analysis(subparsers):
     parser_create_analysis = subparsers.add_parser(
          "create-analysis",
          formatter_class=argparse.ArgumentDefaultsHelpFormatter,
          help="Create MERFISH analysis directory.")

     parser_create_analysis_req = parser_create_analysis.add_argument_group("required inputs")
     parser_create_analysis_req.add_argument("--data-set-name",
                                         type=str,
                                         required=True,
                                         help="MERFISH dataset name.")

     parser_create_analysis_req.add_argument("--codebook-name",
                                     type=str,
                                     required=True,
                                     help="Codebook file name.")

     parser_create_analysis_req.add_argument("--data-organization-name",
                                     type=str,
                                     required=True,
                                     help="Data organization file name.")

     parser_create_analysis_req.add_argument("--microscope-parameter-name",
                                     type=str,
                                     required=True,
                                     help="A jason file contains the microscope settings.")

     parser_create_analysis_req.add_argument("--microscope-chromatic-aberration-name",
                                     type=str,
                                     required=True,
                                     help="A pickle file contains the profile for correct chromatic abberation.")

     parser_create_analysis_req.add_argument("--position-name",
                                     type=str,
                                     required=True,
                                     help="A txt file contains the global positions for each FOV.")
     
     parser_create_analysis_opt = parser_create_analysis.add_argument_group("optional inputs")
     parser_create_analysis_opt.add_argument("--data-home",
                                     type=str,
                                     default=None,
                                     help="Base directory for raw data.")
     parser_create_analysis_opt.add_argument("--analysis-home",
                                     type=str,
                                     default=None,
                                     help="Base directory for analysis results.")

def add_registration(subparsers):
     parser_registration = subparsers.add_parser(
          "register-images",
          formatter_class=argparse.ArgumentDefaultsHelpFormatter,
          help="Register images to correct abberations.")
    
     parser_registration_req = parser_registration.add_argument_group("required inputs")
     parser_registration_req.add_argument("--data-set-name",
                                         type=str,
                                         required=True,
                                         help="MERFISH dataset name.")

     parser_registration_req.add_argument("--fov",
                                     type=int,
                                     required=True,
                                     help="Field of view index.")

     parser_registration_req.add_argument("--zpos",
                                     type=float,
                                     required=True,
                                     help="Z positions in uM.")

     parser_registration_req.add_argument("--output-name",
                                     type=str,
                                     required=True,
                                     help="Output file name.")

     parser_registration_opt = parser_registration.add_argument_group("optional inputs")
     parser_registration_opt.add_argument("--register-drift",
                                     type=str2bool,
                                     default=True,
                                     help="A boolen variable indicates whether to correct stage drift.")

     parser_registration_opt.add_argument("--ref-frame-index",
                                     type=int,
                                     default=0,
                                     help="Reference frame index for correcting drift.")

     parser_registration_opt.add_argument("--high-pass-filter-sigma",
                                     type=int,
                                     default=3,
                                     help="Low pass sigma for high pass filter prior to registration.")

     parser_registration_opt.add_argument("--register-color",
                                     type=str2bool,
                                     default=True,
                                     help="A boolen variable indicates whether to correct chromatic abberation.")

     parser_registration_opt.add_argument("--register-color-profile",
                                     type=str,
                                     default=None,
                                     help="A pkl file contains chromatic abberation profile.")

     parser_registration_opt.add_argument("--save-fiducials",
                                     type=str2bool,
                                     default=False,
                                     help="A boolen variable indicates whether to save fiducial images.")

def add_preprocessing(subparsers):
     parser_preprocessing = subparsers.add_parser(
          "process-images",
          formatter_class=argparse.ArgumentDefaultsHelpFormatter,
          help="Preprocessing of MERFISH images.")
    
     parser_preprocessing_req = parser_preprocessing.add_argument_group("required inputs")
     parser_preprocessing_req.add_argument("--data-set-name",
                                         type=str,
                                         required=True,
                                         help="MERFISH dataset name.")

     parser_preprocessing_req.add_argument("--fov",
                                     type=int,
                                     required=True,
                                     help="Field of view index.")

     parser_preprocessing_req.add_argument("--zpos",
                                     type=float,
                                     required=True,
                                     help="Z positions in uM.")

     parser_preprocessing_req.add_argument("--warped-images-name",
                                     type=str,
                                     required=True,
                                     help="Warped images file name.")

     parser_preprocessing_req.add_argument("--output-name",
                                     type=str,
                                     required=True,
                                     help="Output file name.")

     parser_preprocessing_opt = parser_preprocessing.add_argument_group("optional inputs")
     parser_preprocessing_opt.add_argument("--high-pass-filter-sigma",
                                     type=int,
                                     default=3,
                                     help="Low pass sigma for high pass filter.")
     parser_preprocessing_opt.add_argument("--low-pass-filter-sigma",
                                     type=int,
                                     default=1,
                                     help="High pass sigma for low pass filter.")

     parser_preprocessing_opt.add_argument("--scale-factor-file",
                                     type=str,
                                     default=None,
                                     help="Normalization scaling factors.")

     parser_preprocessing_opt.add_argument("--log-transform",
                                     type=str2bool,
                                     default=False,
                                     help="Normalize image magnitude by log transform.")

def add_predict_prob(subparsers):
     parser_preprocessing = subparsers.add_parser(
          "predict-prob",
          formatter_class=argparse.ArgumentDefaultsHelpFormatter,
          help="Predict on-bit probability.")

     parser_predict_prob_req = parser_preprocessing.add_argument_group("required inputs")
     parser_predict_prob_req.add_argument("--data-set-name",
                                          type=str,
                                          required=True,
                                          help="MERFISH dataset name.")

     parser_predict_prob_req.add_argument("--fov",
                                          type=int,
                                          required=True,
                                          help="Field of view index.")

     parser_predict_prob_req.add_argument("--zpos",
                                          type=float,
                                          required=True,
                                          help="Z positions in uM.")

     parser_predict_prob_req.add_argument("--processed-images-name",
                                          type=str,
                                          required=True,
                                          help="Processed images file name.")

     parser_predict_prob_req.add_argument("--model-name",
                                          type=str,
                                          required=True,
                                          help="Model file name.")

     parser_predict_prob_req.add_argument("--output-name",
                                          type=str,
                                          required=True,
                                          help="Output file name.")

     parser_predict_prob_req.add_argument("--kernel-size",
                                          type=int,
                                          default=3,
                                          required=True,
                                          help="Kernel size.")

def add_decoding(subparsers):
     parser_decoding = subparsers.add_parser(
          "decode-images",
          formatter_class=argparse.ArgumentDefaultsHelpFormatter,
          help="Decode MERFISH images.")
    
     parser_decoding_req = parser_decoding.add_argument_group("required inputs")
     parser_decoding_req.add_argument("--data-set-name",
                                      type=str,
                                      required=True,
                                      help="MERFISH dataset name.")

     parser_decoding_req.add_argument("--fov",
                                     type=int,
                                     required=True,
                                     help="Field of view index.")

     parser_decoding_req.add_argument("--zpos",
                                     type=float,
                                     required=True,
                                     help="Z positions in uM.")

     parser_decoding_req.add_argument("--decoding-images-name",
                                     type=str,
                                     required=True,
                                     help="Decoding images file name.")

     parser_decoding_req.add_argument("--output-name",
                                     type=str,
                                     required=True,
                                     help="Output file name.")

     parser_decoding_opt = parser_decoding.add_argument_group("optional inputs")
     parser_decoding_opt.add_argument("--max-cores",
                                     type=int,
                                     default=5,
                                     help="Max number of CPU cores.")

     parser_decoding_opt.add_argument("--border-size",
                                     type=int,
                                     default=80,
                                     help="Number of pixels to be ignored from the border.")

     parser_decoding_opt.add_argument("--magnitude-threshold",
                                     type=float,
                                     default=0,
                                     help="Threshold for pixel magnitude.")

     parser_decoding_opt.add_argument("--distance-threshold",
                                     type=float,
                                     default=0.65,
                                     help="Threshold between pixel trace and closest barcode.")

def add_train_psm(subparsers):
     parser_train_psm = subparsers.add_parser(
          "train-psm",
          formatter_class=argparse.ArgumentDefaultsHelpFormatter,
          help="Train Pixel Score Machine.")
    
     parser_train_psm_req = parser_train_psm.add_argument_group("required inputs")
     parser_train_psm_req.add_argument("--data-set-name",
                                      type=str,
                                      required=True,
                                      help="MERFISH dataset name.")

     parser_train_psm_req.add_argument("--decoded-images-dir",
                                     type=str,
                                     required=True,
                                     help="Directory that contains all the decoded images.")

     parser_train_psm_req.add_argument("--output-name",
                                     type=str,
                                     required=True,
                                     help="Output psm model file name.")

     parser_train_psm_opt = parser_train_psm.add_argument_group("optional inputs")
     parser_train_psm_opt.add_argument("--zpos-num",
                                     type=int,
                                     default=50,
                                     help="Number of zplanes for training PSM model.")

def add_extract_barcodes(subparsers):
    parser_extract_barcodes= subparsers.add_parser(
         "extract-barcodes",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
         help="Extract Barcodes.")
    
    parser_extract_barcodes_req = parser_extract_barcodes.add_argument_group("required inputs")
    parser_extract_barcodes_req.add_argument("--data-set-name",
                                     type=str,
                                     required=True,
                                     help="MERFISH dataset name.")

    parser_extract_barcodes_req.add_argument("--fov",
                                    type=int,
                                    required=True,
                                    help="Field of view index.")

    parser_extract_barcodes_req.add_argument("--zpos",
                                    type=float,
                                    required=True,
                                    help="Z positions in uM.")

    parser_extract_barcodes_req.add_argument("--decoded-images-name",
                                    type=str,
                                    required=True,
                                    help="Decoded image file name.")

    parser_extract_barcodes_req.add_argument("--output-name",
                                    type=str,
                                    required=True,
                                    help="Output barcode file name.")

    parser_extract_barcodes_opt = parser_extract_barcodes.add_argument_group("optional inputs")
    parser_extract_barcodes_opt.add_argument("--psm-name",
                                    type=str,
                                    default=None,
                                    help="Pixel scoring machine name.")

    parser_extract_barcodes_opt.add_argument("--barcodes-per-core",
                                    type=int,
                                    default=10,
                                    help="Number of barcodes to be decoded per core.")

    parser_extract_barcodes_opt.add_argument("--max-cores",
                                    type=int,
                                    default=10,
                                    help="Max number of cores for parallel computing.")

def add_extract_pixel_traces(subparsers):
    parser_extract_pixel_traces= subparsers.add_parser(
         "extract-pixel-traces",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
         help="Extract decoded pixel traces.")
    
    parser_extract_pixel_traces_req = parser_extract_pixel_traces.add_argument_group("required inputs")
    parser_extract_pixel_traces_req.add_argument("--data-set-name",
                                     type=str,
                                     required=True,
                                     help="MERFISH dataset name.")

    parser_extract_pixel_traces_req.add_argument("--fov",
                                    type=int,
                                    required=True,
                                    help="Field of view index.")

    parser_extract_pixel_traces_req.add_argument("--zpos",
                                    type=float,
                                    required=True,
                                    help="Z positions in uM.")

    parser_extract_pixel_traces_req.add_argument("--extracted-barcodes-name",
                                    type=str,
                                    required=True,
                                    help="Extracted barcode file name.")

    parser_extract_pixel_traces_req.add_argument("--processed-images-name",
                                    type=str,
                                    required=True,
                                    help="Processed image file name.")

    parser_extract_pixel_traces_req.add_argument("--decoded-images-name",
                                    type=str,
                                    required=True,
                                    help="Decoded image file name.")

    parser_extract_pixel_traces_req.add_argument("--output-name",
                                    type=str,
                                    required=True,
                                    help="Output barcode file name.")
    
    parser_extract_pixel_traces_opt = parser_extract_pixel_traces.add_argument_group("optional inputs")
    parser_extract_pixel_traces_opt.add_argument("--area-size-threshold",
                                    type=int,
                                    default=5,
                                    help="Threshold for pixel number.")

    parser_extract_pixel_traces_opt.add_argument("--distance-threshold",
                                    type=float,
                                    default=0.65,
                                    help="Threshold for distance to assigned barcode.")

    parser_extract_pixel_traces_opt.add_argument("--magnitude-threshold",
                                    type=float,
                                    default=0,
                                    help="Threshold for pixel magnitude.")
    
def add_export_barcodes(subparsers):
    parser_export_barcodes= subparsers.add_parser(
         "export-barcodes",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
         help="Export Barcodes.")

    parser_export_barcodes_req = parser_export_barcodes.add_argument_group("required inputs")
    parser_export_barcodes_req.add_argument("--data-set-name",
                                     type=str,
                                     required=True,
                                     help="MERFISH dataset name.")

    parser_export_barcodes_req.add_argument("--decoded-barcodes-dir",
                                    type=str, 
                                    required=True,
                                    help="Directory contains the decoded barcodes.")

    parser_export_barcodes_req.add_argument("--output-name",
                                    type=str,
                                    required=True,
                                    help="Output barcode file name.")

def add_filter_barcodes(subparsers):
    parser_filter_barcodes= subparsers.add_parser(
         "filter-barcodes",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
         help="Filter Barcodes.")
    
    parser_filter_barcodes_req = parser_filter_barcodes.add_argument_group("required inputs")
    parser_filter_barcodes_req.add_argument("--data-set-name",
                                     type=str,
                                     required=True,
                                     help="MERFISH dataset name.")

    parser_filter_barcodes_req.add_argument("--exported-barcodes-name",
                                    type=str, 
                                    required=True,
                                    help="A list of decoded barcode file names.")

    parser_filter_barcodes_req.add_argument("--output-name",
                                    type=str,
                                    required=True,
                                    help="Output barcode file name.")

    parser_filter_barcodes_opt = parser_filter_barcodes.add_argument_group("optional inputs")
    parser_filter_barcodes_opt.add_argument("--fov-num",
                                    type=int,
                                    default=20,
                                    help="Number of fov used for estimating barcode thresholds.")

    parser_filter_barcodes_opt.add_argument("--keep-blank-barcodes",
                                    type=str2bool,
                                    default=True,
                                    help="A boolen variable indicates whether to keep blank barcodes.")

    parser_filter_barcodes_opt.add_argument("--mis-identification-rate",
                                    type=float,
                                    default=0.05,
                                    help="Mis identification rate for barcodes.")

    parser_filter_barcodes_opt.add_argument("--min-area-size",
                                    type=int,
                                    default=1,
                                    help="Min area size.")

def add_segmentation(subparsers):
    parser_segmentation = subparsers.add_parser(
         "segmentation",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
         help="Feature Segmentation.")
    
    parser_segmentation_req = parser_segmentation.add_argument_group("required inputs")
    parser_segmentation_req.add_argument("--data-set-name",
                                     type=str,
                                     required=True,
                                     help="MERFISH dataset name.")
    
    parser_segmentation_req.add_argument("--fov",
                                    type=int,
                                    required=True,
                                    help="Field of view index.")

    parser_segmentation_req.add_argument("--zpos",
                                    type=float,
                                    required=True,
                                    help="Z positions in uM.")
    
    parser_segmentation_req.add_argument("--warped-images-name",
                                    type=str,
                                    required=True,
                                    help="Warped images file name.")

    parser_segmentation_req.add_argument("--feature-name",
                                    type=str,
                                    required=True,
                                    help="Feature name for segmentation.")
    
    parser_segmentation_req.add_argument("--output-name",
                                    type=str,
                                    required=True,
                                    help="Output segmentated image.")

    parser_segmentation_opt = parser_segmentation.add_argument_group("optional inputs")
    parser_segmentation_opt.add_argument("--diameter",
                                    type=int,
                                    default=150,
                                    help="Average diameter of the features.")

    parser_segmentation_opt.add_argument("--model-type",
                                    type=str,
                                    default="nuclei",
                                    help="Model type for cell pose.")

    parser_segmentation_opt.add_argument("--gpu",
                                    type=bool,
                                    default=False,
                                    help="A bool variable indicates whether to use GPU")

def add_extract_features(subparsers):
    parser_extract_features = subparsers.add_parser(
         "extract-features",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
         help="Extract features from segmented images.")
    
    parser_extract_features_req = parser_extract_features.add_argument_group("required inputs")
    parser_extract_features_req.add_argument("--data-set-name",
                                     type=str,
                                     required=True,
                                     help="MERFISH dataset name.")

    parser_extract_features_req.add_argument("--fov",
                                    type=int,
                                    required=True,
                                    help="Field of view index.")

    parser_extract_features_req.add_argument("--zpos",
                                    type=float,
                                    required=True,
                                    help="Zplane position.")

    parser_extract_features_req.add_argument("--segmented-images-name",
                                    type=str,
                                    required=True,
                                    help="Segmented image file name.")

    parser_extract_features_req.add_argument("--output-name",
                                    type=str,
                                    required=True,
                                    help="Output feature name.")

def add_export_features(subparsers):
    parser_export_features = subparsers.add_parser(
         "export-features",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
         help="Export features.")
    
    parser_export_features_req = parser_export_features.add_argument_group("required inputs")
    parser_export_features_req.add_argument("--data-set-name",
                                     type=str,
                                     required=True,
                                     help="MERFISH dataset name.")

    parser_export_features_req.add_argument("--segmented-features-dir",
                                     type=str, 
                                     required=True,
                                     help="Extracted feature file names.")

    parser_export_features_req.add_argument("--output-name",
                                     type=str,
                                     required=True,
                                     help="Output file name.")

    parser_export_features_opt = parser_export_features.add_argument_group("optional inputs")
    parser_export_features_opt.add_argument("--buffer-size",
                                     type=int,
                                     default=15,
                                     help="Buffer size for connecting features between zplanes.")

def add_filter_features(subparsers):
    parser_filter_features = subparsers.add_parser(
         "filter-features",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
         help="Filter features.")
    
    parser_filter_features_req = parser_filter_features.add_argument_group("required inputs")
    parser_filter_features_req.add_argument("--data-set-name",
                                     type=str,
                                     required=True,
                                     help="MERFISH dataset name.")

    parser_filter_features_req.add_argument("--exported-features-name",
                                     type=str, 
                                     required=True,
                                     help="Exported feature file names.")

    parser_filter_features_req.add_argument("--output-name",
                                     type=str,
                                     required=True,
                                     help="Output file name.")

    parser_filter_features_opt = parser_filter_features.add_argument_group("optional inputs")
    parser_filter_features_opt.add_argument("--min-zplane",
                                     type=int,
                                     default=3,
                                     help="Min zplane.")

    parser_filter_features_opt.add_argument("--border-size",
                                     type=int,
                                     default=70,
                                     help="Number of pixels are considered to be border.")

def add_barcode_assignment(subparsers):
    parser_barcode_assignment = subparsers.add_parser(
        "assign-barcodes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Assign barcodes to features.")
    
    parser_barcode_assignment_req = parser_barcode_assignment.add_argument_group("required inputs")
    parser_barcode_assignment_req.add_argument("--data-set-name",
                                     type=str,
                                     required=True,
                                     help="MERFISH dataset name.")

    parser_barcode_assignment_req.add_argument("--exported-barcodes-name",
                                     type=str,
                                     required=True,
                                     help="Exported barcode file name.")

    parser_barcode_assignment_req.add_argument("--exported-features-name",
                                     type=str,
                                     required=True,
                                     help="Exported feature file name.")

    parser_barcode_assignment_req.add_argument("--output-name",
                                     type=str,
                                     required=True,
                                     help="Output file name.")

    parser_barcode_assignment_opt = parser_barcode_assignment.add_argument_group("optional inputs")
    parser_barcode_assignment_opt.add_argument("--max-cores",
                                     type=int,
                                     default=1,
                                     help="Max number of CPU cores.")

    parser_barcode_assignment_opt.add_argument("--buffer-size",
                                     type=float,
                                     default=0,
                                     help="Buffer size.")

def add_export_gene_feature_matrix(subparsers):
    parser_export_gene_feature = subparsers.add_parser(
        "export-gene-feature-matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Export gene feature matrix.")
    
    parser_export_gene_feature_req = parser_export_gene_feature.add_argument_group("required inputs")
    parser_export_gene_feature_req.add_argument("--data-set-name",
                                     type=str,
                                     required=True,
                                     help="MERFISH dataset name.")

    parser_export_gene_feature_req.add_argument("--barcodes-name",
                                     type=str,
                                     required=True,
                                     help="Barcode file name.")

    parser_export_gene_feature_req.add_argument("--features-name",
                                     type=str,
                                     required=True,
                                     help="Feature file name.")

    parser_export_gene_feature_req.add_argument("--output-name",
                                     type=str,
                                     required=True,
                                     help="Output file name.")

    parser_export_gene_feature_opt = parser_export_gene_feature.add_argument_group("optional inputs")
    parser_export_gene_feature_opt.add_argument("--max-cores",
                                     type=int,
                                     default=1,
                                     help="Max number of CPU cores.")

def add_estimate_bit_error(subparsers):
    parser_estimate_bit_err = subparsers.add_parser(
        "estimate-bit-error",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Estimate bit errors.")
    
    parser_estimate_bit_err_req = parser_estimate_bit_err.add_argument_group("required inputs")
    parser_estimate_bit_err_req.add_argument("--data-set-name",
                                     type=str,
                                     required=True,
                                     help="MERFISH dataset name.")

    parser_estimate_bit_err_req.add_argument("--pixel-traces-dir",
                                     type=str,
                                     required=True,
                                     help="Directory that contains extracted pixel traces.")

    parser_estimate_bit_err_req.add_argument("--output-name",
                                     type=str,
                                     required=True,
                                     help="Output file name.")

def str2bool(v):
     ## adapted from the answer by Maxim at
     ## https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
     if v.lower() in ('yes', 'true', 't', 'y', '1'):
          return(True)
     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
          return(False)
     else:
          raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
     parse_args()
