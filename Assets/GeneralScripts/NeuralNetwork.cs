using UnityEngine;
using System.Collections;
using MathNet.Numerics;
using LA = MathNet.Numerics.LinearAlgebra;
using DBL = MathNet.Numerics.LinearAlgebra.Double;

public class NeuralNetwork : MonoBehaviour
{
	// ------------------------------------ Properties ------------------------------------
	public bool TrainingPhase;
	public int numIterations;
	public int numInputs;
	public int numHidden;
	public int numOutputs;
	public int numHiddenLayers;
	public float allowedError;
	public float learningRate;
	public float momentum;
	private int numTestCases;

	private LA.Matrix<float> inputs;
	private LA.Matrix<float> ihWeights;
	private LA.Matrix<float> ihBiases;
	private LA.Matrix<float> ihOutputs;
	private LA.Matrix<float> hoWeights;
	private LA.Matrix<float> hoBiases;
	private LA.Matrix<float> outputs;	// holds the current output of the Net

	// ------------------------------------ Initialization ------------------------------------
	void Start ()
	{
		ihWeights = LA.Matrix<float>.Build.Random (numInputs, numHidden);
		hoWeights = LA.Matrix<float>.Build.Random (numHidden, numOutputs);
		ihBiases = LA.Matrix<float>.Build.Random (1, numHidden);
		hoBiases = LA.Matrix<float>.Build.Random (1, numOutputs);
	}

	// ------------------------------------ Getters ------------------------------------
	public LA.Matrix<float> GetOutputWeights(){
		return hoWeights;
	}
	
	public LA.Matrix<float> GetHiddenWeights(){
		return ihWeights;
	}


	// ------------------------------------ Methods ------------------------------------
	private float sigmoid (float x)
	{
			if (x < -45.0f)
					return 0.0f;
			else if (x > 45.0f)
					return 1.0f;
			else
					return 1 / (1 + Mathf.Exp(-x));
	}

	private float hyperTan (float x)
	{
			if (x < -10.0f) 
					return -1.0f;
			else if (x > 10.0f)
					return 1.0f;
			else
					return (float)Trig.Tanh (x);
	}

	private float sampleMean(LA.Vector<float> sample){
		float mean = 0.0f;

		foreach (var elm in sample)
			mean += (float)elm;

		return mean / sample.Count;
	}

	private float sampleStandardDeviation(LA.Vector<float> sample, float mean){
		float variance = 0.0f;

		foreach (var elm in sample){
			float difference = (float)elm - mean;
			variance += difference * difference;
		}

		return Mathf.Sqrt (variance / (sample.Count - 1));
	}

	private void Normalization(LA.Matrix<float> table){
		int numColumns = table.ColumnCount;
		LA.Vector<float> normColumn;

		for (int i = 0; i < numColumns; i++) {
			float mean = sampleMean(table.Column(i));
			float sd = sampleStandardDeviation(table.Column(i), mean);

			normColumn = (table.Column(i)).Subtract(mean);
			normColumn = normColumn.Divide(sd);
			table.SetColumn(i, normColumn);
		}
	}

	public float CostFunction(LA.Matrix<float> targets, LA.Matrix<float> actual){
		LA.Matrix<float> predicted;
		LA.Vector<float> columnSums;

		predicted = actual.Subtract (targets);
		predicted = predicted.PointwisePower (2);
		predicted = predicted.Divide ((float)(2*numTestCases));

		columnSums = predicted.RowSums();

		float cost = 0.0f;
		for (int i = 0; i < numTestCases; i++)
						cost += columnSums [i];

		return cost;
	}

	// ---------------------------- Feed-Forward Part ----------------------------
	public LA.Matrix<float> ComputeOutputs(LA.Matrix<float> newInputs){
		LA.Matrix<float> ihSums;
		LA.Matrix<float> hoSums;
		inputs = newInputs;

		ihSums = inputs.Multiply (ihWeights);
		ihOutputs = ihSums.Add (ihBiases);

		for (int i = 0; i < numHidden; i++)
			ihOutputs [0, i] = sigmoid(ihOutputs [0, i]);

		hoSums = ihOutputs.Multiply(hoWeights);
		outputs = hoSums.Add (hoBiases);

		for (int i = 0; i < numOutputs; i++)
			outputs [0, i] = hyperTan(outputs [0, i]);

		return outputs;
	}

	// ---------------------------- Back-Propagation Part----------------------------

	// returns the deltas in the different layers
	public ArrayList ComputeDeltas(LA.Matrix<float> targetOutputs, float eta, float alpha){
		LA.Matrix<float> oGrads;
		LA.Matrix<float> hGrads;
		ArrayList deltas = new ArrayList();

		// Compute output gradient
		LA.Matrix<float> hyperTanDerivative = (outputs.Negate()).Add(1);
		hyperTanDerivative = hyperTanDerivative.PointwiseMultiply (outputs.Add (1));
		oGrads = (targetOutputs.Subtract (outputs)).PointwiseMultiply(hyperTanDerivative);

		// Compute hidden gradient
		LA.Matrix<float> grads_weights_sums = hoWeights.TransposeAndMultiply (oGrads);
		LA.Matrix<float> sigmoidDerivative = (ihOutputs.Negate ()).Add (1);
		sigmoidDerivative = ihOutputs.PointwiseMultiply (sigmoidDerivative);
		hGrads = (grads_weights_sums.Transpose()).PointwiseMultiply (sigmoidDerivative);

		LA.Matrix<float> outer_sum = ihOutputs.TransposeThisAndMultiply (oGrads);
		LA.Matrix<float> hidden_sum = inputs.TransposeThisAndMultiply (hGrads);

		deltas.Add	(oGrads.Multiply (eta)); 		// oDeltaBias
		deltas.Add	(outer_sum.Multiply (eta));		// oDelta
		deltas.Add	(hGrads.Multiply (eta));		// hDeltaBias
		deltas.Add	(hidden_sum.Multiply (eta));	// hDelta

		return deltas;
	}

	public void LearningPhase(ArrayList inputCases, ArrayList targetCases, float target_cost){
		numTestCases = targetCases.Count / numOutputs;
		//Debug.Log (targetCases);
		// convert input ArrayLists to Matrices
		float[] temp_targets = (float[])targetCases.ToArray (typeof(float));

		LA.Matrix<float> targets = LA.Matrix<float>.Build.Dense(numTestCases, numOutputs, temp_targets);

		float[] temp_inputs = (float[])inputCases.ToArray (typeof(float));
		LA.Matrix<float> inputs = LA.Matrix<float>.Build.Dense(numTestCases, numInputs, temp_inputs);
		//Normalization (inputs);

		LA.Matrix<float> actual = LA.Matrix<float>.Build.Dense (numTestCases, numOutputs, 0);
		float current_cost = CostFunction (targets, actual);

		ArrayList deltas; // store hidden/output layer bias deltas and weights deltas

		LA.Matrix<float> currInput;
		LA.Matrix<float> currTarget;

		for(int i = 1; i <= numIterations && current_cost > target_cost; i++){
			// Acumulate deltas (Needed for weight updates)
			// Given in the same order by UpdateWeights
			LA.Matrix<float> deltaOutputBias = LA.Matrix<float>.Build.Dense (1, numOutputs, 0);
			LA.Matrix<float> deltaOutput = LA.Matrix<float>.Build.Dense (numHidden, numOutputs, 0);
			
			LA.Matrix<float> deltaHiddenBias = LA.Matrix<float>.Build.Dense (1, numHidden, 0);
			LA.Matrix<float> deltaHidden = LA.Matrix<float>.Build.Dense (numInputs, numHidden, 0);

			for (int m = 0; m < numTestCases; m++) {
				currInput = (inputs.Row(m)).ToRowMatrix();
				currTarget = (targets.Row(m)).ToRowMatrix();
				outputs = ComputeOutputs(currInput);

				actual.SetRow(m, outputs.Row(0));
				deltas = ComputeDeltas(currTarget, learningRate, momentum);

				deltaOutputBias = deltaOutputBias.Add((LA.Matrix<float>)deltas[0]);
				deltaOutput = deltaOutput.Add((LA.Matrix<float>)deltas[1]);
				deltaHiddenBias = deltaHiddenBias.Add((LA.Matrix<float>)deltas[2]);
				deltaHidden = deltaHidden.Add((LA.Matrix<float>)deltas[3]);
			}
			// UpdateWeights
			hoBiases = hoBiases.Add (deltaOutputBias.Divide(numTestCases));
			hoWeights = hoWeights.Add (deltaOutput.Divide(numTestCases));
			ihBiases = ihBiases.Add (deltaHiddenBias.Divide(numTestCases));
			ihWeights = ihWeights.Add (deltaHidden.Divide(numTestCases));

			// Compute new Cost
			current_cost = CostFunction(targets, actual);
		}
		Debug.Log(current_cost);
	}

	// Use this for initialization
	/*
	
	// Update is called once per frame
	void Update () {
		
	}*/
}
