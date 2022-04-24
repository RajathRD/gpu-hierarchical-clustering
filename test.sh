# Generate correct testcase output
python unittest/gen_correct_output.py
# Run the CPU version
./src/cpu_clustering
# Compare outputs
while IFS="" read -r p || [ -n "$p" ]
do
  diff unittest/correct_outputs/$p unittest/cpu_outputs/$p
done < unittest/tests.txt