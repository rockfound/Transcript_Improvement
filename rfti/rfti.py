import argparse
import glob
import os
import shutil
from pathlib import Path
from typing import Any, Dict, TypeAlias

import cv2
import pandas as pd
from deepface import DeepFace


def clear_outputs(directory_path: Path) -> None:
    """clears the all files and directories in the specified {directory_path}

    Args:
        directory_path (Path): path to remove all files and directories from
    """
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"failed to remove {file_path} : {e}")
    return


def get_msec(time: str) -> int:
    """Calculates elapsed time in milliseconds given a timestamp of the form H:M:S

    Args:
        time (str): Timestamp in H:M:S: format to be converted to a msecs
    Returns:
        1000 * (S + 60 * M + 3600 * hour)

    """
    time_split = time.split(":")
    hour = int(time_split[0])
    minute = int(time_split[1])
    sec = int(time_split[2])
    return 1000 * (sec + 60 * minute + 3600 * hour)


def get_frame_msec(msec: int, capture: cv2.VideoCapture, frame_id: int, images_path: Path) -> None:
    """given {msec}, {capture}, extracts the frame corresponding to time {msec} from {capture}

    Args:
        msec (int): time in milliseconds
        capture (cv2.VideoCapture): OpenCV VideoCapture object
        frame_id (image): frame_id coresponding to the frame captured at time {msec}
        image_path (string): path of output image
    """
    capture.set(cv2.CAP_PROP_POS_MSEC, msec + 1)
    (ret, frame) = capture.read()
    file_path = images_path / f"ut_{frame_id}_msec_{int(msec)}.png"
    cv2.imwrite(str(file_path), frame)
    return


def make_directory(path: Path, clear_directory: bool = True) -> None:
    """Creates a directory from  a given path

    Args:
        path (string): path of the directory to be created
        clear_directory(bool) : flag to clear all the files and directories contained in {path}
    """
    try:
        os.mkdir(path)
    except FileExistsError as error:
        print("directory already exists", type(error).__name__, "-", error)
        if clear_directory:
            if clear_directory:
                print(f"Removing images in {path}")
                clear_outputs(path)
    except Exception as error:
        print("An error occured: ", type(error).__name__, "-", error)
    return


FaceArea: TypeAlias = Dict[Any, Any]


def get_corners(face_area: FaceArea) -> list[tuple[int, int]]:
    """returns a list containing the top left and bottom right corners of the face
        [(top_left x, top_left y), (bottom_right x, bottom_right y)]

    Args:
        face_area (dictionary): should be of the format {x: x coordinate of top left, y: y coordiante of top left corner, h: height, w: width}
    """
    top_left = (face_area["x"], face_area["y"])
    bottom_right = (face_area["x"] + face_area["w"], face_area["y"] + face_area["h"])
    return [top_left, bottom_right]


def save_with_rectangles(source_image: Path, corners: list[list[tuple[int, int]]], located_faces_path: Path) -> None:
    """Saves an image in ./located_faces_path with the rangles drawn around the located faces

    Args:
        source_image (path): path to source image
        corners (list): _description_
        located_faces_path (path): path to target directory
    """
    image = cv2.imread(str(source_image))
    for corner in corners:
        image = cv2.rectangle(image, corner[0], corner[1], (0, 255, 0), 3)
        image_stem = source_image.stem
        image_path = located_faces_path / f"{image_stem}_rectangles.png"
        cv2.imwrite(str(image_path), image)


def count_faces(faces: list, confidence: int = 0) -> int:
    """Count faces with confidince greater than {confidence}

    Args:
        faces (_type_): _description_
        confidence (int, optional): _description_. Defaults to 0.
    """
    return len([face for face in faces if face["confidence"] > confidence])


def generate_verification_images(
    source_image_directory: Path, target_image_directory: Path, image_names: list[str], speaker_num: list[int]
) -> None:
    """Places images in the correct speaker folder.

    Args:
        source_image_directory (path): source image directory
        target_image_directory (path): target image directory
        image_name (list): list of image names used in speaker identification
        speaker_num (list): assigned speaker number
    """
    for image_name, speaker in zip(image_names, speaker_num):
        target_image_name = f"{image_name.split('.')[0]}_rectangles.png"
        target_speaker_directory = target_image_directory / f"speaker_{speaker}"
        source_face_path = source_image_directory / f"{image_name[:-4]}_rectangles.png"
        target_face_path = target_speaker_directory / target_image_name
        shutil.copy(source_face_path, target_face_path)
    return


def get_speaker_from_path(paths: list[Path]) -> tuple[list[str], list[str], list[str]]:
    """Returns speaker number from an image path.

    Args:
        paths (list): list containing image paths
    """
    image_stems = [path.stem for path in paths]
    speakers = [path.parent.name.split("_")[-1] for path in paths]
    ut_ids = [image_stem.split("_")[1] for image_stem in image_stems]
    msecs = [image_stem.split("_")[3] for image_stem in image_stems]
    return (speakers, ut_ids, msecs)


def generate_frames(
    ut_df: pd.DataFrame, images_path: Path, video_path: Path, clear_directory: bool = True
) -> pd.DataFrame:
    """Extracts the center frame corresponding to each utterance based on a timestamp H:M:S.

    Args:
        ut_df (pd.DataFrame): utterances dataframe
        images_path (Path): path to images
        video_path (Path): path to video
        clear_directory (Bool, optional): If True removes all files in {images_path} from previous runs with the same input filename. Defaults to True.

    Returns:
        pd.DataFrame: updated utterances dataframe with additional columns "first_frame_msec", "center_frame_msec"
    """

    print("Extracting Frames")
    ut_df["first_frame_msec"] = [get_msec(timestamp) for timestamp in ut_df["timestamp"]]
    ut_df["center_frame_msec"] = ut_df["first_frame_msec"].rolling(2).mean().shift(-1)
    # set value for last frame at 1 millisecond after last timestamp
    ut_df.iloc[-1, -1] = ut_df.iloc[-1, -2] + 1
    ut_df["center_frame_msec"] = [int(x) for x in ut_df["center_frame_msec"]]

    # extract the center frame for each utterance and save the corresponding image
    make_directory(images_path, clear_directory)
    capture = cv2.VideoCapture(str(video_path))
    ut_df["frame"] = ut_df.apply(
        lambda x: get_frame_msec(x["center_frame_msec"], capture, x["ut_id"], images_path), axis=1
    )
    capture.release()
    return ut_df


def glob_images(images_path: Path) -> list[Path]:
    """generates a list of all images in a drictory sorted chronologically by time they occured in the video.

    Images should be named in the formate ut_{ut_id}_msce_{frame_time_in_ms}.png
    to sort by ut_id use image_paths.sort(key=lambda x: x.stem.split("_")[1])

    Args:
        images_path (str | os.PathLike): path to images folder

    Returns:
        list[os.PathLike]: a list of images contained in the directory {images_path}
    """
    glob_path = images_path / "*.png"
    image_paths = [Path(filename) for filename in glob.glob(str(glob_path))]
    image_paths.sort(key=lambda x: x.stem.split("_")[3])
    return image_paths


def detect_faces(
    ut_df: pd.DataFrame,
    images_path: Path,
    base_path: Path,
    detector_name: str = "retinaface",
    clear_directory: bool = True,
) -> pd.DataFrame:
    """Given a path to the directory containing the extracted frames this function detects faces
    and saves the frames with rectangels drawn around the detected faces in ./detected_faces

    Args:
        ut_df (pd.DataFrame): _description_
        images_path (Path):the path to the directory containing the frames extracted from the video corresponding
        to the transcript
        base_path (Path): Path to where all file are stored corresponding to the output filename
        detector_name (str, optional): model to be used for detection see DeepFace documentation for a list of choices. Defaults to "retinaface".
        clear_directory (bool, optional): Clear the located_directory path. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    # read in the image names and sort them chronologically
    image_paths = glob_images(images_path)

    # detect faces in previously saved images and save them with bounding boxes
    print("Detecting Faces")
    detector_name = "retinaface"
    faces_extracted = [
        DeepFace.extract_faces(str(image), detector_backend=detector_name, enforce_detection=False)
        for image in image_paths
    ]
    face_count = [count_faces(faces) for faces in faces_extracted]
    face_corners = [[get_corners(face["facial_area"]) for face in faces] for faces in faces_extracted]
    ut_df["image_name"] = [path.name for path in image_paths]
    ut_df["face_count"] = face_count

    located_faces_path = base_path / "detected_faces"
    make_directory(located_faces_path, clear_directory)
    for _, (image, corners) in enumerate(zip(image_paths, face_corners)):
        save_with_rectangles(image, corners, located_faces_path)
    return ut_df


def compare_faces(
    ut_df,
    images_path: Path,
    base_path: Path,
    detector_name: str = "retinaface",
    model_name: str = "ArcFace",
    clear_directory: bool = False,
) -> pd.DataFrame:
    """Identifies unique speakers and labels all faces that match each unique speakers.

    Args:
        ut_df (DataFrame): dataframe containing utterance data
        images_path (Path): path to the images directory
        base_path (Path): path to the current working directory
        detector_name (str, optional): model used for face detection. Defaults to "retinaface".
        model_name (str, optional): model used for face comparrison. Defaults to "ArcFace".
        clear_directory (bool, optional): clears the images directory. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    print("Comparing Faces")
    image_paths = glob_images(images_path)
    face_count = ut_df["face_count"]

    match_verification = [0] * int(len(image_paths))
    speaker_nums = [-1] * int(len(image_paths))
    speaker_id = 0
    model_name = "ArcFace"
    for i, (image_1, speaker_num_1, face_count_1) in enumerate(zip(image_paths, speaker_nums, face_count)):
        if face_count_1 == 1 and speaker_num_1 == -1:
            for j, (image_2, speaker_num_2, face_count_2) in enumerate(
                zip(image_paths[i:], speaker_nums[i:], face_count[i:])
            ):
                if face_count_2 == 1 and speaker_num_2 == -1:
                    # print(str(i) + ',' + str(j))
                    same_face = DeepFace.verify(
                        img1_path=image_1,
                        img2_path=image_2,
                        enforce_detection=False,
                        detector_backend=detector_name,
                        model_name=model_name,
                    )
                    if same_face["verified"]:
                        speaker_nums[i + j] = speaker_id
                        match_verification[i + j] = same_face
            speaker_id = speaker_id + 1

    # Save iamges in speaker directory for verification
    ut_df["image_name"] = [path.name for path in image_paths]
    ut_df["face_count"] = face_count
    ut_df["speaker_num"] = speaker_nums
    ut_df["image_path"] = image_paths
    target_image_path = base_path / "verification_images"
    located_faces_path = base_path / "detected_faces"
    make_directory(target_image_path)
    for speaker in ut_df["speaker_num"].unique():
        make_directory(target_image_path / f"speaker_{speaker}", clear_directory=False)

    generate_verification_images(located_faces_path, target_image_path, ut_df["image_name"], ut_df["speaker_num"])
    return ut_df


def generate_new_transcript(ut_df: pd.DataFrame, verification_images_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the file structure in {verification_images_path} and generates a new transcript based on the directory \
        that eacht image is contained in

    Args:
        ut_df (pd.DataFrame): utterances dataframe 
        verification_images_path (Path): path to verification images

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ut_df is the original utterance dataframe, new_ut_df is the DataFrame \
            corresponding to the new utterances
    """
    test_set_image_paths = glob.glob(str(verification_images_path / "**" / "*.png"), recursive=True)
    test_set_image_paths = [Path(img_path) for img_path in test_set_image_paths]
    (new_speaker, ut_id, msec) = get_speaker_from_path(test_set_image_paths)
    new_speaker_df = pd.DataFrame({"new_speaker": new_speaker, "ut_id": ut_id, "msec": msec})
    new_speaker_df = new_speaker_df.sort_values(by="msec", ignore_index=True)
    ut_df["verified_speaker_num"] = new_speaker_df["new_speaker"].tolist()
    ut_df["ut_id_check"] = new_speaker_df["ut_id"].tolist()
    ut_df["new_speaker"] = ut_df["verified_speaker_num"].tolist()

    new_ut_df = pd.DataFrame()
    new_ut_counter = 0
    new_ut_id = []
    new_text = []
    new_timestamp = []
    new_speaker = []
    combined = []
    og_ut = []
    y_stop = ut_df.shape[0] - 1
    for x in ut_df.index.to_list()[:-1]:
        y = x
        text = ut_df.loc[x, "text"]
        # the try is to get the first element processed this needs to have the correct error caught (out of bounds for checking) and allow all others to halt execution
        try:
            not_same_as_previous = ut_df.loc[x, "new_speaker"] != ut_df.loc[x - 1, "new_speaker"]
        except KeyError as err:
            not_same_as_previous = True
        same_as_next = ut_df.loc[x, "new_speaker"] == ut_df.loc[x + 1, "new_speaker"]
        if not_same_as_previous and same_as_next:
            new_timestamp.append(ut_df.loc[x, "timestamp"])
            new_speaker.append(ut_df.loc[x, "new_speaker"])
            new_ut_id.append(new_ut_counter)
            while y < y_stop and ut_df.loc[y, "new_speaker"] == ut_df.loc[y + 1, "new_speaker"]:
                text = text + "  " + ut_df.loc[y + 1, "text"]
                y = y + 1
            new_text.append(text)
            combined.append("True")
            og_ut.append(ut_df.loc[x, "ut_id"])
        elif not_same_as_previous:
            new_timestamp.append(ut_df.loc[x, "timestamp"])
            new_speaker.append(ut_df.loc[x, "new_speaker"])
            new_text.append("  " + text)
            new_ut_id.append(new_ut_counter)
            combined.append("False")
            og_ut.append(ut_df.loc[x, "ut_id"])
        new_ut_counter = new_ut_counter + 1

    new_ut_df = pd.DataFrame(
        {
            "ut_id": new_ut_id,
            "text": new_text,
            "speaker": new_speaker,
            "timestamp": new_timestamp,
            "combined": combined,
            "original_ut": og_ut,
        }
    )
    return ut_df, new_ut_df


def main(transcript_path_in: str, video_path_in: str, output_filename_in: str) -> None:
    transcript_path = Path(transcript_path_in)
    base_path = Path.cwd() / transcript_path.stem

    video_path = Path(video_path_in)
    output_path = base_path / output_filename_in
    images_path = base_path / "images"

    # create the base_path directory
    make_directory(base_path)
    print(transcript_path)
    # read in the utterances in from transcript path
    ut_df = pd.read_csv(transcript_path)

    # generate frames for analysis
    ut_df = generate_frames(ut_df, images_path, video_path)

    # locate faces and save images with rectagles drawn around faces
    ut_df = detect_faces(ut_df, images_path, base_path)

    # Compare each face against all faces with all faces not yet assigned a speaker
    ut_df = compare_faces(ut_df, images_path, base_path)

    # Pause while user verifies speaker assignment
    input(
        "Verify that the directories in `./verification_images` contain only one speaker. " \
        "Place an unidentified speaker images in `./verification_images/speaker-1/`. "  \
        "You may correct speaker assignment by moving an image to the correct directory or " \
        "a new directory of the form `./verification_images/speaker_n` for some integer n." \
        "Press enter to continue."

    )

    print("Generating New Transcript")
    verification_images_path = base_path / "verification_images"
    ut_df, new_ut_df = generate_new_transcript(ut_df, verification_images_path)

    print(base_path / f"original_{transcript_path.stem}_df.pkl")
    new_ut_df.to_csv(str(Path(output_path)))


if __name__ == "__main__":  # Name of transcript
    parser = argparse.ArgumentParser(description="A python script to improve old zoom transcripts")
    parser.add_argument(
        "transcript_filename",
        type=str,
        help="Name of the transcript to improve.  Should reside in the same directory as transcript_improvement.py",
    )
    parser.add_argument("video_path", type=str, help="The absolut path of the source video for transcription.")
    parser.add_argument("--output_filename", type=str, help="The name of the file to output the improved transcript to")
    args = parser.parse_args()
    # if --output_filename is not passed then fall back to improved_{transcript_filename}.csv
    if args.output_filename is None:
        args.output_filename = f"improved_{Path(args.transcript_filename).stem}.csv"

    main(args.transcript_filename, args.video_path, args.output_filename)
