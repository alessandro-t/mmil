import os
import requests
import sys
import tarfile

def download(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total/1000), \
                                                             1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50-done)))
                sys.stdout.flush()
    sys.stdout.write('\n')

if __name__ == '__main__':
    prefix = 'data'
    if not os.path.exists('data/aclImdb'):
        url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        fname = url.split('/')[-1].strip()
        fname = os.path.join(prefix)
        os.makedirs(fname, exist_ok=True)
        print('Downloading {}...'.format(fname))
        download(url, os.path.join(prefix, fname))
        print('Done.')
        current_dir = os.getcwd()
        os.chdir(prefix)
        print('Extracting {}...'.format(fname))
        tar = tarfile.open(fname)
        tar.extractall()
        tar.close()
        print('Done.')
        os.chdir(current_dir)
        os.remove(os.path.join(prefix,fname)) 
