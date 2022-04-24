# Generate correct testcase output
python unittest/gen_correct_output.py
# Run the CPU version
./src/cpu_clustering
# Compare outputs
while read p; do
  echo "$p"
done < unittest/tests.txt