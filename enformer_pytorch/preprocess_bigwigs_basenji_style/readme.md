## 1: Convert 131k long sequences from sequences.bed from link to basenji_barnyard mouse data

samtools faidx /net/noble/vol3/user/ssontha2/pinglay_noble/2025_ssontha2_longcontextDNAdesign/bin/basenji-designer/tutorials/data/yash_data/mm10.fa
cut -f1,2 /net/noble/vol3/user/ssontha2/pinglay_noble/2025_ssontha2_longcontextDNAdesign/bin/basenji-designer/tutorials/data/yash_data/mm10.fa.fai > /net/noble/vol3/user/ssontha2/pinglay_noble/2025_ssontha2_longcontextDNAdesign/results/yash/11.01.2025_finetuneenformer/mm10.genome

awk 'BEGIN{
FS=OFS="\t";
while((getline < "/net/noble/vol3/user/ssontha2/pinglay_noble/2025_ssontha2_longcontextDNAdesign/results/yash/11.01.2025_finetuneenformer/mm10.genome")>0){sz[$1]=$2}
}
{
mid = int(($2+$3)/2); # center
half = 196608/2; # 98,304
start = mid - half;
end = mid + half;
if (start < 0) start = 0;
if (end > sz[$1]) end = sz[$1];
print $1, start, end, $4
}' /net/noble/vol3/user/ssontha2/pinglay_noble/2025_ssontha2_longcontextDNAdesign/data/10.16.2025_finetuneenformer_files/basenji_barnyard/data/mouse/sequences.bed > /net/noble/vol3/user/ssontha2/pinglay_noble/2025_ssontha2_longcontextDNAdesign/data/10.16.2025_finetuneenformer_files/basenji_data_ourfiles/sequences_196608.bed

## 2: Run basenji_data with --restart option activated using sequwnces_bed.file within the -o directory with 196k long sequences. I modified the code such that you pass in the length of the sequence at -l and the crop on each end in -c and this will hold consistent throughout the script (before basenji_data_write appended extra -c length sequences again weirdly...)

python basenji_data_original.py \
 --restart \
 -l 196608 \
 -w 128 \
 -c 40960 \
 --local -p 8 \
 -o /net/noble/vol3/user/ssontha2/pinglay_noble/2025_ssontha2_longcontextDNAdesign/data/10.16.2025_finetuneenformer_files/basenji_data_ourfiles/basenjI_data_modfied_out \
 /net/noble/vol3/user/ssontha2/pinglay_noble/2025_ssontha2_longcontextDNAdesign/bin/basenji-designer/tutorials/data/yash_data/mm10.fa \
 /net/noble/vol3/user/ssontha2/pinglay_noble/2025_ssontha2_longcontextDNAdesign/bin/basenji-designer/tutorials/data/yash_data/sud_atac_wigs.txt

## 3: Run my modified tfr_to_numpy to get numpy arrays from the output tfRecords from step 2

python tfr_to_numpy.py \
 --data_dir "/net/noble/vol3/user/ssontha2/pinglay_noble/2025_ssontha2_longcontextDNAdesign/data/10.16.2025_finetuneenformer_files/basenji_data_ourfiles/basenjI_data_modfied_out" \
 --output_dir "/net/noble/vol3/user/ssontha2/pinglay_noble/2025_ssontha2_longcontextDNAdesign/data/10.16.2025_finetuneenformer_files/basenji_data_ourfiles/tfrecord_to_numpy"
