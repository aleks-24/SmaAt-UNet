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
from collections import defaultdict
from root import ROOT_DIR
from utils import data_loader_precip, dataset_precip, data_loader_precip, dataset_hybrid
from models import unet_precip_regression_lightning as unet_regr

def get_binary_metrics(model, test_dl, loss="mse", denormalize=True, thresholds = [0.5], mask_empty = True):
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
          factor = 80.54

      epsilon = 1e-6
      counts = defaultdict(lambda: defaultdict(int))
      for threshold in thresholds:
          counts[str(threshold)]['total_tp'] = 0
          counts[str(threshold)]['total_fp'] = 0
          counts[str(threshold)]['total_tn'] = 0
          counts[str(threshold)]['total_fn'] = 0
          
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
          
          for threshold in thresholds:
                  
              y_pred_mask = y_pred_adj > threshold
              y_true_mask = y_true_adj > threshold
        
              # also add extra mask to remove blank pixels
              if mask_empty:
                  map_mask = np.load("mask.npy")
                  map_mask = map_mask.astype(np.uint8)
                  map_mask = torch.from_numpy(map_mask).to('cuda').squeeze()
                  map_mask = map_mask.unsqueeze(0).repeat(y_pred_mask.shape[0], 1, 1) #repeat for batch size
                  y_pred_mask = y_pred_mask[map_mask==1]
                  y_true_mask = y_true_mask[map_mask==1]
              
              y_pred_mask = y_pred_mask.cpu()
              y_true_mask = y_true_mask.cpu()
              tn, fp, fn, tp = np.bincount(y_true_mask.view(-1) * 2 + y_pred_mask.view(-1), minlength=4)
              counts[str(threshold)]['total_tp'] += tp
              counts[str(threshold)]['total_fp'] += fp
              counts[str(threshold)]['total_tn'] += tn
              counts[str(threshold)]['total_fn'] += fn

      mse_image = loss_denorm / len(test_dl)
      mse_pixel = mse_image / torch.numel(y_true)

      results = defaultdict(lambda: defaultdict(float))
      for threshold in thresholds:
          total_tp = counts[str(threshold)]['total_tp']
          total_fp = counts[str(threshold)]['total_fp']
          total_tn = counts[str(threshold)]['total_tn']
          total_fn = counts[str(threshold)]['total_fn']
          
          print(f"TP: {total_tp}")
          print(f"FP: {total_fp}")
          print(f"TN: {total_tn}")
          print(f"FN: {total_fn}")
          
          # get metrics
          results[str(threshold)]['precision'] = total_tp / (total_tp + total_fp + epsilon)
          results[str(threshold)]['recall'] = total_tp / (total_tp + total_fn + epsilon)
          results[str(threshold)]['accuracy'] = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + epsilon)
          p = results[str(threshold)]['precision']
          r = results[str(threshold)]['recall']
          results[str(threshold)]['f1'] = 2 * p * r / (p + r + epsilon)
          results[str(threshold)]['csi'] = total_tp / (total_tp + total_fn + total_fp + epsilon)
          results[str(threshold)]['far'] = total_fp / (total_tp + total_fp + epsilon)
          results[str(threshold)]['hss'] = (total_tp * total_tn - total_fn * total_fp) / ((total_tp + total_fn) * (total_fn + total_tn) + (total_tp + total_fp) * (total_fp + total_tn) + epsilon)
          
    return mse_pixel.item(), mse_image.item(), results

def print_binary_metrics(model, data_file, thresholds):
    test_dl = data_file
    mse_pixel, mse_image, results = get_binary_metrics(model, test_dl, loss="mse",
                                                                                    denormalize=True, thresholds=thresholds, mask_empty = False)
    mse_pixel_mask, mse_image_mask, mask_results = get_binary_metrics(model, test_dl, loss="mse",
                                                                                    denormalize=True, thresholds=thresholds, mask_empty = True)
    for threshold in thresholds:
        precision = results[str(threshold)]['precision']
        recall = results[str(threshold)]['recall']
        accuracy = results[str(threshold)]['accuracy']
        f1 = results[str(threshold)]['f1']
        csi = results[str(threshold)]['csi']
        far = results[str(threshold)]['far']
        hss = results[str(threshold)]['hss']

        precision_mask = mask_results[str(threshold)]['precision']
        recall_mask = mask_results[str(threshold)]['recall']
        accuracy_mask = mask_results[str(threshold)]['accuracy']
        f1_mask = mask_results[str(threshold)]['f1']
        csi_mask = mask_results[str(threshold)]['csi']
        far_mask = mask_results[str(threshold)]['far']
        hss_mask = mask_results[str(threshold)]['hss']
        
        print(
            f"MSE (pixel): {mse_pixel}, MSE (image): {mse_image}, precision: {precision}, recall: {recall}, accuracy: {accuracy}, f1: {f1}, csi: {csi}, far: {far}, hss: {hss}")
        print("Masked values")
        print(
            f"MSE (pixel): {mse_pixel_mask}, MSE (image): {mse_image_mask}, precision: {precision_mask}, recall: {recall_mask}, accuracy: {accuracy_mask}, f1: {f1_mask}, csi: {csi_mask}, far: {far_mask}, hss: {hss_mask}")
    
    return [mse_pixel, mse_image, results], [mse_pixel_mask, mse_image_mask, mask_results]


def get_model_losses(model_folder, data_file, loss, denormalize):
    # Save it to a dict that can be saved (and plotted)
    test_losses = dict()
    test_losses_masked = dict()
    

    models = [m for m in os.listdir(model_folder) if ".ckpt" in m]
    # dataset = dataset_precip.precipitation_maps_masked_h5(
    dataset = dataset_hybrid.precipitation_maps_h5_nodes(
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

        thresholds = [0.5, 10, 20]

        if is_first:
          for threshold in thresholds:
            print(str(int(threshold*100)))
            test_losses[f"binary_{str(int(threshold*100))}"] = []
            test_losses_masked[f"binary_{str(int(threshold*100))}"] = []
          is_first = False

        binary_loss, binary_loss_masked = print_binary_metrics(model, test_dl, thresholds=thresholds)

        for threshold in thresholds:
            metrics = list(binary_loss[-1][str(threshold)].values())
            metrics_masked = list(binary_loss_masked[-1][str(threshold)].values())
        
            test_losses[f"binary_{str(int(threshold*100))}"].append([threshold, name, False, binary_loss[:2]] + metrics)
            test_losses_masked[f"binary_{str(int(threshold*100))}"].append([threshold, name, True, binary_loss[:2]] + metrics_masked)
            print(losses_to_csv(test_losses, (results_folder / "res_35.csv")))
            print(losses_to_csv(test_losses_masked, (results_folder / "res_35_masked.csv")))
        
    return

def losses_to_csv(losses_dict, path):
    csv = "threshold, name, masked, mse (pixel), mse (image), precision, recall, accuracy, f1, csi, far, hss\n"
    for key, losses in losses_dict.items():
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
    model_folder = ROOT_DIR / "comparison/Smaat"
    data_file = (
        ROOT_DIR / "data" / "precipitation" / "hybrid_train_test_2014-2023_input-length_12_img-ahead_6_rain-threshold_35.h5"
    )
    results_folder = ROOT_DIR / "results" / "SmaAt_UNet"

    test_losses = dict()
    test_losses_masked = dict()
    get_model_losses(model_folder, data_file, loss, denormalize)

    

