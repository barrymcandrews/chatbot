from typing import List
from datetime import datetime
import boto3
import click
import os
import glob
from tqdm import tqdm
import inquirer


BUCKET_NAME = 'sagemaker-chatbot-builds'

s3 = boto3.client('s3')
s3_resource = boto3.resource('s3')
bucket = s3_resource.Bucket(BUCKET_NAME)


def get_builds() -> List[str]:
    paginated_results = s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET_NAME)
    all_objects = [e['Key'] for p in paginated_results for e in p['Contents']]
    folders = list(set([o.split('/')[0] for o in all_objects if len(o.split('/')) > 1]))
    folders.sort()
    return folders


def upload_build(name=('build_' + datetime.now().strftime("%d_%m_%Y-%H_%M_%S"))):
    print("Uploading build '" + name + "'")
    for root, dirs, files in os.walk("build", topdown=True):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        files[:] = [f for f in files if not f.startswith('.') and not f.endswith('.tmp')]
        for f in files:
            r = '/'.join(root.split('/')[1:])
            path = os.path.join(r, f)
            bucket.upload_file('build/' + path, name + '/' + path)
    print("Upload complete.")


def download_build(name):
    print("Downloading build '" + name + "'")
    for obj in bucket.objects.filter(Prefix=name):
        remote_path = obj.key
        root_path = '/'.join(obj.key.split('/')[1:])
        local_path = 'build/' + root_path
        if not os.path.exists(os.path.dirname(local_path)):
            os.makedirs(os.path.dirname(local_path))

        bucket.download_file(remote_path, local_path)
    print("Download complete.")


def delete_build(name):
    if name not in get_builds():
        print('Build not found.')
        return
    print("Deleting build '" + name + "'")
    bucket.objects.filter(Prefix=name).delete()


#
# Command Line Interface
#

@click.group()
def builds():
    pass


@builds.command(name='list')
def ls():
    for b in get_builds():
        print(b)


@builds.command()
@click.argument('name', type=str, required=False)
def put(name):
    if name in get_builds():
        if not inquirer.confirm("There is already a build with that name. Do you want to replace it?", default=False):
            return

    upload_build(name)


@builds.command()
@click.argument('name', type=str, required=False)
def get(name):
    if os.path.exists('build/model'):
        pass
    if name is None:
        q = [inquirer.List('name', message='Which model would you like to download?', choices=get_builds())]
        name = inquirer.prompt(q)['name']

    download_build(name)



@builds.command()
@click.argument('name', type=str, required=False)
def delete(name):
    if name is None:
        name = inquirer.list_input('Which model would you like to delete?', choices=get_builds())

    if inquirer.confirm("This will permanently delete your model. Do you want to continue?", default=False):
        delete_build(name)
