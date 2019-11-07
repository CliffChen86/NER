import argparse
import torch
import os 

import logging

import asyncio
import aiohttp.web

import nermodel.version
from nermodel.model_wrapper import ModelWrapper
from nermodel.http_handler import HTTPHandler
from nermodel.globals import RespType


parser = argparse.ArgumentParser(description='A simple NER serving application by using PyTorch and aiohttp.')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=512,
                    help='batch size for training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--use-cuda', type=bool,default=False,
                    help='enables cuda')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--use-crf', action='store_true',
                    help='use crf')

parser.add_argument('--mode', type=str, default='train',
                    help='train mode or test mode')

parser.add_argument('--save', type=str, default='/model/lstm_crf.pth',
                    help='path to save the final model')
parser.add_argument('--save-epoch', action='store_true',
                    help='save every epoch')
parser.add_argument('--data', type=str, default='dataset',
                    help='location of the data corpus')

parser.add_argument('--word-ebd-dim', type=int, default=300,
                    help='number of word embedding dimension')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout')
parser.add_argument('--lstm-hsz', type=int, default=512,
                    help='BiLSTM hidden size')
parser.add_argument('--lstm-layers', type=int, default=2,
                    help='biLSTM layer numbers')
parser.add_argument('--l2', type=float, default=0.005,
                    help='l2 regularization')
parser.add_argument('--clip', type=float, default=.5,
                    help='gradient clipping')
parser.add_argument('--cudaID', type=int, default=1,
                    help='cuda-device-id')
parser.add_argument('--vocab-path', type=str, default='/model/vocab.pkl',
                    help='cuda-device-id')
parser.add_argument('--default-response-type', type=str, default='json',
                    help='The default response type if request does not specify. Should in (msgpack, json).')
parser.add_argument('--unix-socks-path', type=str, default='',
                    help='Unix socks file path')
parser.add_argument('--port', type=int, default=8000,
                    help='http port')
parser.add_argument('--host', type=str, default='0.0.0.0',
                    help='http host')

args = parser.parse_args()

 
torch.manual_seed(args.seed)



__all__ = ['entry_point']
logger = logging.getLogger('ner-serving')

def _str2bool(s: str) -> bool:
    return s.lower() in ['t', 'true', 'on', '1']

def entry_point():
    print(args)
    path = args.unix_socks_path
    try:
        port = args.port
    except (TypeError, ValueError):
        port = None
    host = args.host

    model = ModelWrapper(args)

    handler = HTTPHandler(model=model,
                          default_response_type=RespType[
                              args.default_response_type.upper()])
    app = aiohttp.web.Application()
    app.add_routes([aiohttp.web.post('/', handler.post)])

    app.on_startup.append(model.startup)

    # aiohttp.web.run_app(app, port=port, path=path, host=host)
    aiohttp.web.run_app(app, port=port, host=host)
