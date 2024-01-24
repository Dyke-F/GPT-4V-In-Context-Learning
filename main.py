import json
import os
import time
from getpass import getpass
from pprint import pprint

from pathlib import Path
import fsspec
import hydra

from dotenv import load_dotenv
from omegaconf import OmegaConf

from dataset import GPT4VEvalDataset, num_tokens_from_messages
from knn_dataset import GPT4VKNNDataset
# from multi_image_knn_dataset import GPT4MultiImageKNNNDataset

from vision import GPT4V

os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(config_path="./config/CRC100K/knn", config_name="zero_shot", version_base="1.3") # TODO: change config_path to your config folder
def main(cfg):
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or getpass(
        f"Enter a valid OpenAI API key: "
    )
    assert os.environ["OPENAI_API_KEY"].startswith("sk-"), f"Invalid OpenAI API key"

    data_cfg = cfg.data
    model_cfg = cfg.model
    user_args = cfg.user_args

    mode = "zero_shot" if data_cfg.num_shots == 0 else "multi_shot"

    if data_cfg.num_shots == 0:
        assert (
            data_cfg.show_bbox == False
        ), "show_bbox can only be used with num_shots > 0"

    with fsspec.open(user_args.system_prompt_path, mode="r") as f:
        system_prompt = f.read()

    with fsspec.open(user_args.user_query_path, mode="r") as f:
        user_query = f.read()

    # if we are running in random sampling mode use the default GPT4VEvalDataset
    if not hasattr(data_cfg, "dataset_vectors_path"):
        dataset = GPT4VEvalDataset(
            datafile_path=data_cfg.datafile_path,
            use_only=data_cfg.use_only,
            label_replacements=data_cfg.label_replacements,
            use_tiles=data_cfg.use_tiles if hasattr(data_cfg, "use_tiles") else False,
        )
        print("Running in random-sampling mode.")

    # if we have a dataset_vectors_path provided we are running in KNN mode
    # use the Phikon vector embeddings for #-shot example sampling 
    else:
        dataset = GPT4VKNNDataset(
            datafile_path=data_cfg.datafile_path,
            use_only=data_cfg.use_only,
            label_replacements=data_cfg.label_replacements,
            dataset_vectors_path=data_cfg.dataset_vectors_path,
            use_tiles=data_cfg.use_tiles if hasattr(data_cfg, "use_tiles") else False,
            most_similar_last=data_cfg.most_similar_last if hasattr(data_cfg, "most_similar_last") else False,
            take_random=data_cfg.take_random if hasattr(data_cfg, "take_random") else False,
            #take_random=False,
        )
        print("Running in KNN-sampling mode.")

    dataset.pprint_self()
    print("# dataframe entries: ", len(dataset.data))

    model_kwargs = OmegaConf.to_container(
        model_cfg.model_kwargs, resolve=True
    )  # TODO: implement usage

    model = GPT4V(
        model_name=model_cfg.model_name,
        system_prompt=system_prompt,
        user_query=user_query,
        mode=mode,
        dataset=dataset,
        img_quality=model_cfg.img_quality,
        model_kwargs=model_kwargs,
        show_bbox=data_cfg.show_bbox,
    )

    batch = []
    samples = []
    batch_size = data_cfg.batch_size

    try:
        for idx, sample in enumerate(
            dataset(data_cfg.num_shots, show_bbox=data_cfg.show_bbox), start=1
        ):
            
            # this is to re-run all experiments in case GPT-4V was interrupted due to server issues, connections etc ...
            # as the API is not stateful, we can do this in a new session
            if hasattr(data_cfg, "samples"):
                if sample["fname"] not in data_cfg.samples:
                    print(f"Skipping sample #{idx} ...")
                    continue
            
            print(f"Running sample #{idx} ...")
            # TODO: implement batched mode -> currently doesnt reliably work on ChatCompletions API, check forum for fixes from time to time
            # TODO: the below snippet will not work, but can be used as a template later as soon as batch mode is enabled for ChatCompletions and GPT Vision 
            if user_args.batched:
                samples.append(sample) # to index later
                messages = model.preprocess_inputs(sample)
                batch.append(messages)
                
                if idx % batch_size == 0:
                    print(f"Batch collection of length {len(batch)} completed. Running model ...")
                    responses = model.run_model(batch)
                    
                    for response in responses:
                        print("Response: ", response.choices[0].message.content)
                        print("Sample Label: ", sample["label"])

                        sample = samples[response.choices.index]

                        model.prepare_for_save(response, sample, data_cfg, user_args)
                    batch.clear() # thats safer than accumulating and list slicing
                    time.sleep(15)
            else:
                if user_args.verbose:
                    run_verbose(model, sample, model_cfg, data_cfg, user_args)
                else:
                    model(sample)

            if user_args.debug:
                break

            time.sleep(10)

        if user_args.batched:
            if batch:
                print(f"Final batch collection of length {len(batch)} completed. Running model ...")
                responses = model.run_model(batch)
                for response in responses:
                    model.prepare_for_save(response, sample, data_cfg, user_args)

    except KeyboardInterrupt:
        print("Keyboard Interrupt Detected. Exiting with saving ...")
    except Exception as e:
        print("An error occurred:", str(e))

    finally:
        existing_runs_len = len(list(Path(data_cfg.save_path).rglob("*.json")))
        num_run = existing_runs_len + 1
        save_path = f"{data_cfg.save_path}/knn_result_{cfg.mode}_run{num_run}" 
        model.save_conversation(save_path)
        print("Saved_output to: ", save_path)
        print("COMPLETED.")


def run_verbose(model, sample, model_cfg, data_cfg, user_args):
    messages = model.preprocess_inputs(sample)

    # uncomment the below to save the messages to a json file for debugging before running the model
    # with open("messages.json", "w") as f:
    #     json.dump(messages, f, indent=4)
    # exit()

    num_tokens = num_tokens_from_messages(
        messages, model=model_cfg.model_name, img_quality=model_cfg.img_quality
    )

    print(f"APPROXIMATELY USING #{num_tokens} tokens.")
    response = model.run_model(messages)
    print("Sample name: ", sample["fname"])
    print("Response: ", response.choices[0].message.content)
    print("Sample Label: ", sample["label"])
    model.prepare_for_save(response, sample, data_cfg, user_args)
    print("=" * 120)


if __name__ == "__main__":
    main()
