# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('./deep_text_recognition_benchmark')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./i-can-read-379204-ce1c5c2f12f5.json"

import uvicorn
import torch
import pickle
import configparser
import requests
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile
from transformModel import transform_data
from preprocessImage import crop_image, preprocess_image
from deep_text_recognition_benchmark import demo
from deep_text_recognition_benchmark.model import Model
from deep_text_recognition_benchmark.utils import CTCLabelConverter, AttnLabelConverter
from deep_text_recognition_benchmark.dataset import RawDataset, AlignCollate
# from deep_text_recognition_benchmark.modules import transformation

app = FastAPI(max_request_size=1024*1024*1024)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './saved_models/pretrained/best_accuracy.pth'

dir_name = "words"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


class Options:
    def __init__(self):
        self.num_fiducial = 20
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100
        self.rgb = False
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 256
        self.character = '가각간갇갈감갑값갓강갖같갚갛개객걀걔거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀귓규균귤그극근글긁금급긋긍기긴길김깅깊까깍깎깐깔깜깝깡깥깨꺼꺾껌껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꾼꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냇냉냐냥너넉넌널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐댓더덕던덜덟덤덥덧덩덮데델도독돈돌돕돗동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿링마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몬몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭘뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벨벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브븐블비빌빔빗빚빛빠빡빨빵빼뺏뺨뻐뻔뻗뼈뼉뽑뿌뿐쁘쁨사삭산살삶삼삿상새색샌생샤서석섞선설섬섭섯성세섹센셈셋셔션소속손솔솜솟송솥쇄쇠쇼수숙순숟술숨숫숭숲쉬쉰쉽슈스슨슬슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓴쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액앨야약얀얄얇양얕얗얘어억언얹얻얼엄업없엇엉엊엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷옹와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡잣장잦재쟁쟤저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쩔쩜쪽쫓쭈쭉찌찍찢차착찬찮찰참찻창찾채책챔챙처척천철첩첫청체쳐초촉촌촛총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칫칭카칸칼캄캐캠커컨컬컴컵컷케켓켜코콘콜콤콩쾌쿄쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택터턱턴털텅테텍텔템토톤톨톱통퇴투툴툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔팝패팩팬퍼퍽페펜펴편펼평폐포폭폰표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홈홉홍화확환활황회획횟횡효후훈훌훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘?!.,()'
        self.Transformation = 'TPS'
        self.FeatureExtraction = 'ResNet'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'CTC'
        self.saved_model = model_path
        self.image_folder = "./words"
        self.workers = 0
        self.batch_size = 64


config = configparser.ConfigParser()
config.read('config.ini')

host = config.get('server', 'host')
port = config.getint('server', 'port')


@app.get("/")
async def root():
    return {"message": f"Server running on {host}:{port}"}


@app.post('/api/v1/menu/extract')
async def extract_text(file: UploadFile):
    results = []
    menu = preprocess_image(file)
    crop_image(menu)

    opt = Options()

    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)
    # model = torch.nn.DataParallel(model).to(device)
    # model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    transform_data(model, './saved_models/cafe/transform.pth')

    ## Load the .pkl file
    with open('./saved_models/new_model.pkl', 'rb') as f:
        state_dict = pickle.load(f)

    ## When I use just .pth file
    # torch.save(model.state_dict(), new_model_path)
    # model.load_state_dict(torch.load(new_model_path))
    #
    # state_dict = torch.load(new_model_path)
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     name = k.replace('module.', '')  # remove 'module.' from key name
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)

    # resize the Prediction.bias tensor to match the size in the checkpoint
    if 'Prediction.bias' in model.state_dict():
        new_bias_size = model.state_dict()['Prediction.bias'].size()
        old_bias_size = state_dict['Prediction.bias'].size()

        if old_bias_size != new_bias_size:
            state_dict['module.Prediction.bias'] = state_dict['module.Prediction.bias'][:new_bias_size[0]]
            model.load_state_dict(state_dict, strict=False)

    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()

    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)

            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

                result = pred
                # print(f'{img_name:25s}\t{pred:25s}')
                results.append(result)
    print(results)
    return results


if __name__ == '__main__':
    uvicorn.run(app, host=host, port=port)
