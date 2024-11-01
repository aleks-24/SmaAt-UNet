import torch
from torch import nn, triangular_solve
import matplotlib.pyplot as plt
import numpy as np
import os
from os import path
import pickle
from tqdm import tqdm
import math
from pathlib import Path

from root import ROOT_DIR
from utils import data_loader_precip, dataset_precip, data_loader_precip, dataset_hybrid
from models import unet_precip_regression_lightning as unet_regr

def get_binary_metrics(model, test_dl, loss="mse", denormalize=True, threshold=0.5):
    with torch.no_grad():
      cuda = torch.device("cuda")
      model.eval()  # or model.freeze()?
      model.to(cuda)

      if loss.lower() == "mse":
          loss_func = nn.functional.mse_loss
      elif loss.lower() == "mae":
          loss_func = nn.functional.l1_loss
      factor = 1
      if denormalize:
          factor = 47.83

      threshold = threshold
      epsilon = 1e-6

      total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0 

      loss_denorm = 0.0

      count = 0
      for input_img, input_2, target_img, target_2 in tqdm(test_dl, leave=False):
          x = input_img
          y_true = target_img
          count += 1
          x = x.to(cuda)
          y = input_2.to(cuda)
          y_true = y_true.to(cuda).squeeze()
          y_true = y_true

          y_pred = model(x)
          y_pred = y_pred.squeeze()
          
          # denormalize
          y_pred_adj = y_pred * factor
          y_true_adj = y_true * factor
          # calculate loss on denormalized data
          loss_denorm += loss_func(y_pred_adj, y_true_adj, reduction="sum")

          # convert to mm/h
          y_pred_adj *= 12.0
          y_true_adj *= 12.0
          
          # convert to masks for comparison
          y_pred_mask = y_pred_adj > threshold
          y_true_mask = y_true_adj > threshold
          
          y_pred_mask = y_pred_mask.cpu()
          y_true_mask = y_true_mask.cpu()
          tn, fp, fn, tp = np.bincount(y_true_mask.view(-1) * 2 + y_pred_mask.view(-1), minlength=4)
          total_tp += tp
          total_fp += fp
          total_tn += tn
          total_fn += fn

      mse_image = loss_denorm / len(test_dl)
      mse_pixel = mse_image / torch.numel(y_true)

      print(f"TP: {total_tp}")
      print(f"FP: {total_fp}")
      print(f"TN: {total_tn}")
      print(f"FN: {total_fn}")
      # get metrics
      precision = total_tp / (total_tp + total_fp + epsilon)
      recall = total_tp / (total_tp + total_fn + epsilon)
      accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + epsilon)
      f1 = 2 * precision * recall / (precision + recall + epsilon)
      csi = total_tp / (total_tp + total_fn + total_fp + epsilon)
      far = total_fp / (total_tp + total_fp + epsilon)
      pod = total_tp / (total_tp + total_fn + epsilon)
      hss = (total_tp * total_tn - total_fn * total_fp) / ((total_tp + total_fn) * (total_fn + total_tn) + (total_tp + total_fp) * (total_fp + total_tn) + epsilon)
    return mse_pixel.item(), mse_image.item(), precision, recall, accuracy, f1, csi, far, pod, hss

def print_binary_metrics(model, data_file, threshold=0.5):
    test_dl = data_file
    mse_pixel, mse_image, precision, recall, accuracy, f1, csi, far, pod, hss = get_binary_metrics(model, test_dl, loss="mse",
                                                                                    denormalize=True, threshold=threshold)
    print(
        f"MSE (pixel): {mse_pixel}, MSE (image): {mse_image}, precision: {precision}, recall: {recall}, accuracy: {accuracy}, f1: {f1}, csi: {csi}, far: {far}, pod: {pod}, hss: {hss}")
    return [mse_pixel, mse_image, precision, recall, accuracy, f1, csi, far, pod, hss]


def get_model_losses(model_folder, data_file, loss, denormalize):
    # Save it to a dict that can be saved (and plotted)
    test_losses = dict()
    

    models = [m for m in os.listdir(model_folder) if ".ckpt" in m]
    # dataset = dataset_precip.precipitation_maps_masked_h5(
    dataset = dataset_hybrid.precipitation_maps_h5_kriging(
        in_file=data_file,
        num_input_images=12,
        num_output_images=6, 
        train=False)

    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    is_first = True
    # load the models
    for model_file in models:
        print(model_folder)
        print(model_file)
        model, model_name = unet_regr.SmaAt_UNet, "SmaAt_UNet"
        model = model.load_from_checkpoint(f"{model_folder}/{model_file}")

        name = model_name

        thresholds = [0.5]

        if is_first:
          for threshold in thresholds:
            print(str(int(threshold*100)))
            test_losses[f"binary_{str(int(threshold*100))}"] = []
          is_first = False

        for threshold in thresholds:
          
          binary_loss = print_binary_metrics(model, test_dl, threshold=threshold)
          row = list(binary_loss)
          test_losses[f"binary_{str(int(threshold*100))}"].append([threshold, name] + list(binary_loss))

        
    return test_losses

def losses_to_csv(losses, path):
    csv = "threshold, name, mse (pixel), mse (image), precision, recall, accuracy, f1, csi, far, pod, hss\n"
    for loss in losses:
      row = ",".join(str(l) for l in loss)
      csv += row + "\n"

    with open(path,"w+") as f:
      f.write(csv)

    return csv


if __name__ == '__main__':
    loss = "mse"
    denormalize = True
    # Models that are compared should be in this folder (the ones with the lowest validation error)
    model_folder = ROOT_DIR / "comparison"
    data_file = (
        ROOT_DIR / "data" / "precipitation" / "hybrid_kriging_train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_50_size_64_64.h5"
    )
    results_folder = ROOT_DIR / "results" / "SmaAt_UNet"

    test_losses = dict()
    test_losses = get_model_losses(model_folder, data_file, loss, denormalize)
    print(losses_to_csv(test_losses['binary_50'], (results_folder / "res_50.csv")))

