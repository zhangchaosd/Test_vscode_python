import os

def fuc(path='/Users/zhangchao/Desktop/dzl'):
    print('now parse ', path)
    fs = os.listdir(path)
    if '.DS_Store' in fs:
        print('rm ds')
        os.remove(os.path.join(path,'.DS_Store'))
        fs.remove('.DS_Store')
    videos = []
    # print(fs)
    for f in fs:
        if f[:2]=='Fi' and  f[-4:] == '.mp4':
            videos.append(f)
    for v in videos:
        v=os.path.join(path,v)
        os.system("ffmpeg -i %s %s" %
          (v, v+'.webm'))
        os.remove(v)
        # os.rename(v[:-5]+'2.webm', v)
    folders = [os.path.join(path,f) for f in fs if f[-4:] != '.mp4' and f[-4:] != 'webm' and f[-4:] != '.png']
    # print(folders)
    for f in folders:
        fuc(f)

fuc()