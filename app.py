import streamlit as st
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from predict_page import show_predict_page

show_predict_page()