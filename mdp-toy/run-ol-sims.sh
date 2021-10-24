echo "Beginning Open Loop Simulations..."

declare -a transitions=("advanced"
                        "high-break-low-replace"
                        "low-break-high-replace"
                        "pure-cutspeed"
                        "used-tools"
                        "variable-broken"
                        "variable-replace"
                        "wear-based-reward-high-negative-penalty"
                        "wear-based-reward-low-negative-penalty")

for i in {0..3}
do
    for name in "${transitions[@]}"
    do
        poetry run python src/open-loop.py $name $i &
    done

    echo "Waiting for jobs to finish for state ${i}..."
    wait
    echo "Completed"
done
