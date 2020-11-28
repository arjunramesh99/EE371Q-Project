#!/usr/bin/python3
import shutil
import os
from argparse import ArgumentParser
from pathlib import Path

class DatasetParser:
    '''
        Emotion Map from file
        1: Surprise
        2: Fear
        3: Disgust
        4: Happiness
        5: Sadness
        6: Anger
        7: Neutral
     NOTE: Array indexes from 0-6 in the same order.
    '''

    emotion = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

    def __init__(self, base_path, dest_path):
        self._emo_image_map = {}
        self._emo_image_count = [0] * len(DatasetParser.emotion)        
        self._base_path = Path(base_path)
        self._dest_path = Path(dest_path)
        self._images_path = self._base_path / 'aligned'
        self._img_count = 0

    def _insert_image(self, category, emo_num, img_name, line_count):
        map_obj = {}
        if category not in self._emo_image_map:
            self._emo_image_map[category] = [ [] for i in range(len(DatasetParser.emotion)) ]
        try:
            self._emo_image_map[category][emo_num-1].append(img_name.rsplit('.', 1)[0])
        except IndexError as e:
            print(e)
            print(category, emo_num, img_name, line_count)

    def load_data_map(self, filename, max_files = None):
        line_count = 0;
        with open(self._base_path / filename) as f:
            lines = f.readlines()
            for line in lines:
                img_details, emo_num = line.split()
                category, img_name = img_details.split('_', 1)
                # Insert into emo_map dict
                self._insert_image(category, int(emo_num), img_details, line_count)
                line_count = line_count + 1;
                if max_files is not None and line_count >= max_files:
                    break

            self._img_count = line_count


    def create_file_hierarchy(self, cleardest = False):
        if self._dest_path.exists() and cleardest:
            shutil.rmtree(self._dest_path)

        if not self._dest_path.exists():    
            os.mkdir(self._dest_path)
        count = 0;
        for cat, inst in self._emo_image_map.items():
            cat_path = self._dest_path / cat
            if not cat_path.exists():
                os.mkdir(cat_path)

            for emotion_num in range(len(inst)):
                emo_cat_path = cat_path / str(emotion_num)
                if not emo_cat_path.exists():
                    os.mkdir(emo_cat_path)
                for img_name in inst[emotion_num]:
                    img_filename = img_name + '_aligned.jpg'             
                    shutil.copy(src = self._images_path / img_filename, dst = emo_cat_path)
                    count = count + 1;
            
            print(f"Copied {cat} data")


    def get_img_count(self):
        return self._img_count

    def print_dataset_stats(self):
        print("DATASET STATS\n")
        for cat, inst in self._emo_image_map.items():
            emo_count = [len(inst[i]) for i in range(len(inst))]
            cat_count = sum(emo_count)
            print(f"{cat} Count: {cat_count}")
            for i in range(len(emo_count)):
                print(f"   {DatasetParser.emotion[i]}\t:   {emo_count[i]}".expandtabs(16))
            print("\n")
                        
            #print(self._emo_image_map)

if __name__ == '__main__':
    arg_parser = ArgumentParser(description='Organizes dataset for machine learning model input')
    arg_parser.add_argument('-b', '--basepath', default='./RAF', help= 'Base path for dataset. Default: ./RAF')
    arg_parser.add_argument('-d', '--destpath', required=True)
    arg_parser.add_argument('-m', '--max-files', default=None, type=int, help='Max files to read from base path')
    arg_parser.add_argument('-c', '--clear-dest', action='store_true', help='Specifies whether destpath should be cleared')

    args = arg_parser.parse_args()

    RAF_Parser = DatasetParser(args.basepath, args.destpath)
    RAF_Parser.load_data_map('list_partition_label.txt', max_files = args.max_files)
    RAF_Parser.print_dataset_stats()
    RAF_Parser.create_file_hierarchy(cleardest = args.clear_dest)
