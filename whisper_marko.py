import whisper
import logging
import time
from typing import Any
from concurrent.futures import TimeoutError, ThreadPoolExecutor
from pathlib import Path


# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(model_name: str) -> Any:
    """
    Load the Whisper model.

    :param model_name: The name of the model to load.
    :return: The loaded model object.
    """
    start_time = time.time()
    try:
        model = whisper.load_model(model_name)
        logging.info(f"Model '{model_name}' loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load the model '{model_name}'. Error: {e}")
        raise
    end_time = time.time()
    logging.info(f"Time taken to load model: {end_time - start_time:.2f} seconds")
    return model


def transcribe_audio(model: Any, audio_path: str, timeout: int = 300) -> dict:
    """
    Transcribe the audio file using the Whisper model.

    :param model: The Whisper model.
    :param audio_path: The path to the audio file to transcribe.
    :param timeout: Maximum time allowed for transcription in seconds.
    :return: The transcription result as a dictionary.
    """
    start_time = time.time()
    try:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(model.transcribe, audio_path)
            result = future.result(timeout=timeout)
            logging.info("Transcription completed successfully.")
    except TimeoutError:
        logging.error(f"Transcription timed out after {timeout} seconds.")
        raise
    except Exception as e:
        logging.error(f"Transcription failed. Error: {e}")
        raise
    end_time = time.time()
    logging.info(f"Time taken to transcribe audio: {end_time - start_time:.2f} seconds")
    return result


def save_transcription(result: dict, output_file: str) -> None:
    """
    Save the transcription result to a file.

    :param result: The transcription result dictionary.
    :param output_file: The path to the output file where transcription will be saved.
    """
    try:
        with open(output_file, 'w') as file:
            file.write(result['text'])
        logging.info(f"Transcription saved to '{output_file}'.")
    except Exception as e:
        logging.error(f"Failed to save transcription to '{output_file}'. Error: {e}")
        raise


def main() -> None:
    """
    Main function to load the model, transcribe audio, and save the result.
    """
    try:
        model_name: str = "medium"
        model = load_model(model_name)

        audio_path: str = input("Enter the path to your audio file: ")
        if not Path(audio_path).is_file():
            logging.error(f"Audio file '{audio_path}' does not exist.")
            return

        result = transcribe_audio(model, audio_path)

        output_file: str = "outputfile.txt"
        save_transcription(result, output_file)

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
