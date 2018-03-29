import json

if __name__ == '__main__':
    # Load v7w telling annotation
    v7w_telling = json.load(open('dataset_v7w_telling.json'))

    # Get the mapping from a qa_id to image_id and qa_pair_id in v7w_telling
    qa_id_to_image_id = {}
    qa_id_to_qa_pair_id = {}
    for n in xrange(len(v7w_telling['images'])):
        for i in xrange(len(v7w_telling['images'][n]['qa_pairs'])):
            qa_id_to_image_id[v7w_telling['images'][n]['qa_pairs'][i]['qa_id']] = n
            qa_id_to_qa_pair_id[v7w_telling['images'][n]['qa_pairs'][i]['qa_id']] = i

    # Load v7w telling box grounding annotation
    v7w_telling_answers = json.load(open('dataset_v7w_grounding_annotations/v7w_telling_answers.json'))

    # Add grounding boxes in v7w_telling_answers to v7w_telling
    for item in v7w_telling_answers['boxes']:
        image_id = qa_id_to_image_id[item['qa_id']]
        qa_pair_id = qa_id_to_qa_pair_id[item['qa_id']]
        if 'boxes' in v7w_telling['images'][image_id]['qa_pairs'][qa_pair_id]:
            v7w_telling['images'][image_id]['qa_pairs'][qa_pair_id]['boxes'].append(item)
        else:
            v7w_telling['images'][image_id]['qa_pairs'][qa_pair_id]['boxes'] = []
            v7w_telling['images'][image_id]['qa_pairs'][qa_pair_id]['boxes'].append(item)

    # Show the grounding boxes for each question-answer pair in each image
    for image in v7w_telling['images']:
        for item in image['qa_pairs']:
            print '-------------------'
            print 'image_id: ', item['image_id']
            print 'qa_id: ', item['qa_id']
            print 'question: ', item['question']
            print 'answer: ', item['answer']
            if 'boxes' in item:
                print 'boxes: ', item['boxes']