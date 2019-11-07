## Ner-Service

A http ner-service based on ner-master.

##### Requirements:
    Docker

#### How to use
    run ./tools/build_docker_image.sh
    docker run nerservice:latest


#### How to use on your own dataset:
    train ner-master based on your own dataset
    move your vocab.pkl into datasets filefolder
    move your model save file(.pth) into checkpoints filefolder

