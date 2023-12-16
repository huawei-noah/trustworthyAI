
set -x
while read n_nodes p num_samples batch_size nb_epoch input_dimension hidden_dim exponent_type
do
  for seed in {0..4}
  do
    sbatch run_experiment.sh \
          --n_nodes $n_nodes \
          --p $p \
          --seed $seed \
          --num_samples $num_samples \
          --batch_size $batch_size \
          --nb_epoch $nb_epoch \
          --input_dimension $input_dimension \
          --hidden_dim $hidden_dim \
          --exponent_type $exponent_type
  done
done < configs