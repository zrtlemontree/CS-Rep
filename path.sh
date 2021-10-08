export KALDI_ROOT=## your kaldi path
base_egs=egs/sre16/v2
export PATH=${KALDI_ROOT}/${base_egs}/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:${KALDI_ROOT}/${base_egs}:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export PYTHONOPTIMIZE=1
export PYTHONPATH=`pwd`
export PATH=${KALDI_ROOT}/${base_egs}/utils/run.pl:$PATH
