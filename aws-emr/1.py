import boto.emr
from boto.emr.connection import EmrConnection
from boto.emr.step import StreamingStep
from boto.emr.step import InstallPigStep
from boto.emr.step import PigStep
import threading

access_key_id = 'AKIAI4RIDIOEBKTO622Q'
secret_key = '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'

pig_file = 's3://kagglesenta/pig/convert2fann.pig'
INPUT = 's3://kagglesenta/train.tsv'
OUTPUT = 's3://kagglesenta/output/fann_sentiments.train'
pig_args = ['-p', 'INPUT=%s' % INPUT,
            '-p', 'OUTPUT=%s' % OUTPUT]
log_uri = 's3://kagglesenta/jobflow_logs'

conn = EmrConnection(access_key_id, secret_key)

# step = StreamingStep(name='My wordcount example',
#                       mapper='s3n://elasticmapreduce/samples/wordcount/wordSplitter.py',
#                       reducer='aggregate',
#                       input='s3n://elasticmapreduce/samples/wordcount/input',
#                       output='s3n://out555/output/wordcount_output')
# steps = [step]

pig_step = PigStep('Prepare snt features', pig_file, pig_args=pig_args)
steps = [InstallPigStep(), pig_step]

jobid = conn.run_jobflow(name='Prepare snt features job',
                         log_uri=log_uri,
                         ami_version='latest',
                 		 num_instances=2,
                 		 keep_alive=False,
                         steps=steps)

print 'jobid', jobid

CHECK_STATUS_INT = 10.0
def printStatus():
	status = conn.describe_jobflow(jobid)

	if status != 'COMPLETED'\
		and status != 'FAILED'\
		and status != 'TERMINATED':
		print 'status.state = ', status.state
		t = threading.Timer(CHECK_STATUS_INT, printStatus)
		t.start()

t = threading.Timer(CHECK_STATUS_INT, printStatus)
t.start()

t.join()




