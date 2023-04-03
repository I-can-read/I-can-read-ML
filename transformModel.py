# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('./deep_text_recognition_benchmark')

import torch
import pickle


def transform_data(model, new_model_path):

    # Save the state dictionary to a file
    torch.save(model.state_dict(), new_model_path)

    # Load the state dictionary from the file
    state_dict = torch.load(new_model_path)

    # Serialize the state dictionary using pickle
    with open('./saved_models/new_model.pkl', 'wb') as f:
        pickle.dump(state_dict, f)
