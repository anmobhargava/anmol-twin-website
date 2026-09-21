# Dockerfile (project root)
#
# Container image for the twin's Lambda function. Originally needed for
# packaging torch + sentence-transformers (whose size exceeded Lambda's
# zip/layer limits), but that local embedding model has since been replaced
# with AWS Bedrock Titan Embeddings (an API call, not a bundled model) --
# see vector_store.py's own comment for the full reasoning. Kept as a
# container image regardless (rather than reverting to a zip), since it's
# still a clean, working deployment path and there's no strong reason to
# switch back now that it's set up and working.
#
# Uses AWS's own Lambda base image, which already provides the Lambda
# Runtime Interface Client -- no extra glue code needed to make a
# container image work as a Lambda function.
FROM public.ecr.aws/lambda/python:3.12

# Preserves the EXACT same backend/ + corpus/ sibling layout this repo
# already has locally -- ${LAMBDA_TASK_ROOT} is Lambda's own equivalent of
# "the zip root" from the old approach, and putting backend/ and corpus/
# here as siblings means lambda_handler.py's existing
# `os.path.join(os.path.dirname(__file__), "..", "corpus")` logic keeps
# working completely unchanged, and the dotted handler path
# "backend.lambda_handler.handler" (also unchanged) still resolves
# correctly, since ${LAMBDA_TASK_ROOT} is on sys.path by default.
COPY backend ${LAMBDA_TASK_ROOT}/backend
COPY corpus ${LAMBDA_TASK_ROOT}/corpus

# Installed directly into LAMBDA_TASK_ROOT (rather than a separate layer)
# so rank_bm25/scikit-learn/boto3/etc. are importable the same way as
# backend/'s own code. Much lighter now that torch/sentence-transformers/
# faiss are all gone -- this should build in well under a minute, versus
# the 20+ minutes it took with torch's ~370MB+ download.
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/backend/rag/requirements.txt --target ${LAMBDA_TASK_ROOT}

CMD ["backend.lambda_handler.handler"]