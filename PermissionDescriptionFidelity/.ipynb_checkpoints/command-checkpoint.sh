#!/bin/bash

PERMISSION_TYPE=$1
MODEL_TYPE="BaseModel"
OUTPUT_DIR="../reports/$MODEL_TYPE"

# Create output directory if not exists
mkdir -p $OUTPUT_DIR

python runner.py 	--permission-type $PERMISSION_TYPE \
					--saved-parameters-dir  /home/huseyinalecakir/Security/data/saved-parameters  \
					--saved-data saved-data/emdeddings-sentences-w2i.pickle \
					--saved-reviews saved-data/reviews.pickle \
					--saved-predicted-reviews saved-data/predicted-$PERMISSION_TYPE-reviews.pickle \
					--model-checkpoint saved-models/$MODEL_TYPE-$PERMISSION_TYPE.pt \
					--outdir $OUTPUT_DIR/$PERMISSION_TYPE.out

touch $OUTPUT_DIR/README

COMMIT_ID="$(git rev-parse HEAD)"
echo "Aciklama : " > $OUTPUT_DIR/README
echo "COMMIT ID : $COMMIT_ID"  >> $OUTPUT_DIR/README
