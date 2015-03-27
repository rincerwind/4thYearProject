using UnityEngine;
using System.Collections;
using MathNet.Numerics;
using LA = MathNet.Numerics.LinearAlgebra;
using DBL = MathNet.Numerics.LinearAlgebra.Double;

public class NeuralNetwork : MonoBehaviour
{
	// ------------------------------------ Properties ------------------------------------
	public int numIterations;			// 
	public int numInputs;				// number of neurons in Input Layer
	public int numHidden;				// number of neurons in Hidden Layer
	public int numOutputs;				// number of neurons in Output Layer
	public int numHiddenLayers;			// number of hidden layers
	public float allowedError;			// a threshold of the error
	public float learningRate;
	public float momentum;
	public ArrayList allOutputs;		// stores the outputs of the training set
	public ArrayList allInputs;			// stores the inputs of the training set
	public bool Classification = true;

	private int numTestCases;			
	private ArrayList w_weights;		// stores the weights of the different layers
	private ArrayList w_biases;			// stores the biases of the different layers
	private ArrayList w_outputs;		// stores the outputs of the different layers
	private ArrayList biasDeltas;		// stores the deltas of the biases in the different layers
	private ArrayList weightDeltas;		// stores the deltas of the weights in the different layers

	// ------------------------------------ Initialization ------------------------------------
	void Start ()
	{
		// Init Network
		allOutputs = new ArrayList ();
		allInputs = new ArrayList ();

		w_weights = new ArrayList(numHiddenLayers + 1);
		w_biases = new ArrayList(numHiddenLayers + 1);
		w_outputs = new ArrayList(numHiddenLayers + 2);
		biasDeltas = new ArrayList(numHiddenLayers + 1);
		weightDeltas = new ArrayList(numHiddenLayers + 1);

		initArrayList(ref w_outputs, numHiddenLayers + 2);
		initArrayList(ref biasDeltas, numHiddenLayers + 1);
		initArrayList(ref weightDeltas, numHiddenLayers + 1);
	
		w_weights.Add ( LA.Matrix<float>.Build.Random (numInputs, numHidden) );
		w_biases.Add ( LA.Matrix<float>.Build.Random (1, numHidden) );

		for(int i = 1; i < numHiddenLayers; i++){
			w_weights.Add ( LA.Matrix<float>.Build.Random (numHidden, numHidden) );
			w_biases.Add ( LA.Matrix<float>.Build.Random (1, numHidden) );
		}

		w_weights.Add ( LA.Matrix<float>.Build.Random (numHidden, numOutputs) );
		w_biases.Add ( LA.Matrix<float>.Build.Random (1, numOutputs) );
	}

	// ------------------------------------ Getters ------------------------------------


	// ------------------------------------ Helpers ------------------------------------
	private void initArrayList(ref ArrayList x, int numElms){
		for(int i = 0; i < numElms; i++)
			x.Add(0);
	}

	// ------------------------------------ Step Functions ------------------------------------
	private float sigmoid (float x)
	{
			if (x < -45.0f)
					return 0.0f;
			else if (x > 45.0f)
					return 1.0f;
			else
					return 1 / (1 + Mathf.Exp(-x));
	}

	private LA.Matrix<float> sigmoidOnMatrix (LA.Matrix<float> x, int len)
	{
		for (int i = 0; i < len; i++)
			x[0, i] = sigmoid(x[0, i]);

		return x;
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

	private LA.Matrix<float> hyperTanOnMatrix (LA.Matrix<float> x, int len)
	{
		for (int i = 0; i < len; i++)
			x[0, i] = hyperTan(x[0, i]);

		return x;
	}

	// ------------------------------------ End Step Functions ------------------------------------
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

	// ---------------------------- Cost Functions ----------------------------

	public float CostFunction(LA.Matrix<float> targets, LA.Matrix<float> actual){
		if( Classification == false )
			return LinearCostFunction(targets, actual);

		return LogisticCostFunction(targets,actual);
	}

	private float LinearCostFunction(LA.Matrix<float> targets, LA.Matrix<float> actual){
		LA.Matrix<float> err;
		float cost = 0.0f;

		err = actual.Subtract (targets);
		err = err.PointwisePower (2);

		for (int i = 0; i < numOutputs; i++)
			cost += err [0,i];
		return cost;
	}

	private float LogisticCostFunction(LA.Matrix<float> targets, LA.Matrix<float> actual){
		LA.Matrix<float> err;
		LA.Matrix<float> pos;
		LA.Matrix<float> neg;
		float cost = 0.0f;

		pos = targets.PointwiseMultiply(actual.PointwiseLog());
		neg = targets.Negate().Add(1);
		neg = neg.PointwiseMultiply((actual.Negate().Add(1)).PointwiseLog());

		err = pos.Add(neg);

		for (int i = 0; i < numOutputs; i++)
			cost += err [0,i];
		
		return cost;
	}

	// ---------------------------- Feed-Forward Part ----------------------------

	public LA.Matrix<float> ComputeOutputs(LA.Matrix<float> newInputs){
		LA.Matrix<float> sums;
		//inputs = newInputs;
		w_outputs[0] = newInputs;

		sums = ((LA.Matrix<float>)w_outputs[0]).Multiply ((LA.Matrix<float>)w_weights[0]);
		w_outputs[1] = sums.Add ((LA.Matrix<float>)w_biases[0]);
		w_outputs[1] = sigmoidOnMatrix((LA.Matrix<float>)w_outputs[1], numHidden);

		int i = 1;
		for( i = 1; i < w_weights.Count - 1; i++ ){
			sums = ((LA.Matrix<float>)w_outputs[i]).Multiply ((LA.Matrix<float>)w_weights[i]);
			w_outputs[i+1] = sums.Add ((LA.Matrix<float>)w_biases[i]);
			w_outputs[i+1] = sigmoidOnMatrix((LA.Matrix<float>)w_outputs[i+1], numHidden);
		}

		sums = ((LA.Matrix<float>)w_outputs[i]).Multiply ((LA.Matrix<float>)w_weights[i]);
		w_outputs[i+1] = sums.Add ((LA.Matrix<float>)w_biases[i]);

		if( Classification == false )
			w_outputs[i+1] = hyperTanOnMatrix((LA.Matrix<float>)w_outputs[i+1], numOutputs);
		else
			w_outputs[i+1] = sigmoidOnMatrix((LA.Matrix<float>)w_outputs[i+1], numOutputs);
		
		return (LA.Matrix<float>)w_outputs[i+1];
	}

	// ---------------------------- Back-Propagation Part----------------------------

	// computes the deltas of the different layers
	public void ComputeDeltas2(ref LA.Matrix<float> targetOutputs, float eta, float alpha, 
	                               ref ArrayList biasDeltas, 
	                           	   ref ArrayList weightDeltas){
		LA.Matrix<float> grads;
		LA.Matrix<float> layer_sums;
		LA.Matrix<float> curr_outputs;
		LA.Matrix<float> grads_weights_sums;
		LA.Matrix<float> sigDeriv;
		LA.Matrix<float> hTanDeriv;

		// Compute hidden-output gradients
		curr_outputs = (LA.Matrix<float>)w_outputs[w_outputs.Count-1];

		if( Classification == false ){
			hTanDeriv = (curr_outputs.Negate()).Add(1);
			hTanDeriv = hTanDeriv.PointwiseMultiply (curr_outputs.Add (1));
			grads = (targetOutputs.Subtract (curr_outputs)).PointwiseMultiply(hTanDeriv);
		}
		else{
			sigDeriv = (curr_outputs.Negate()).Add (1);
			sigDeriv = curr_outputs.PointwiseMultiply (sigDeriv);
			grads = (targetOutputs.Subtract (curr_outputs)).PointwiseMultiply(sigDeriv);
		}

		layer_sums = ((LA.Matrix<float>)w_outputs[w_outputs.Count-2]).TransposeThisAndMultiply(grads);

		biasDeltas[biasDeltas.Count - 1] = 
			((LA.Matrix<float>)biasDeltas[biasDeltas.Count - 1]).Add(grads.Multiply(eta));
		weightDeltas[weightDeltas.Count - 1] = 
			((LA.Matrix<float>)weightDeltas[weightDeltas.Count - 1]).Add(layer_sums.Multiply(eta));

		// Compute hidden-hidden gradients
		for(int i = w_outputs.Count - 2; i > 0; i--){
			grads_weights_sums = ((LA.Matrix<float>)w_weights[i]).TransposeAndMultiply (grads);
			curr_outputs = (LA.Matrix<float>)w_outputs[i];

			sigDeriv = (curr_outputs.Negate ()).Add (1);
			sigDeriv = curr_outputs.PointwiseMultiply (sigDeriv);
			grads = (grads_weights_sums.Transpose()).PointwiseMultiply (sigDeriv);
			layer_sums = ((LA.Matrix<float>)w_outputs[i-1]).TransposeThisAndMultiply(grads);

			biasDeltas[i-1] = ((LA.Matrix<float>)biasDeltas[i-1]).Add(grads.Multiply(eta));
			weightDeltas[i-1] = ((LA.Matrix<float>)weightDeltas[i-1]).Add(layer_sums.Multiply(eta));
		}
	}

	public void LearningPhase(ArrayList inputCases, ArrayList targetCases, float target_cost){
		LA.Matrix<float> currInput;
		LA.Matrix<float> currTarget;
		float current_cost = 1000f;
		numTestCases = targetCases.Count / numOutputs;

		// convert input ArrayLists to Matrices
		float[] temp_targets = (float[])targetCases.ToArray (typeof(float));
		LA.Matrix<float> targets = LA.Matrix<float>.Build.Dense(numTestCases, numOutputs, temp_targets);

		float[] temp_inputs = (float[])inputCases.ToArray (typeof(float));
		LA.Matrix<float> inputs = LA.Matrix<float>.Build.Dense(numTestCases, numInputs, temp_inputs);
		//Normalization (inputs);

		int i;
		for(i = 1; i <= numIterations && current_cost > target_cost; i++){
			// Acumulate deltas (Needed for weight updates)
			// Given in the same order by UpdateWeights

			biasDeltas[0] = LA.Matrix<float>.Build.Dense (1, numHidden, 0);
			weightDeltas[0] = LA.Matrix<float>.Build.Dense (numInputs, numHidden, 0);

			for(int l = 1; l < numHiddenLayers; l++ ){
				biasDeltas[l] = LA.Matrix<float>.Build.Dense (1, numHidden, 0);
				weightDeltas[l] = LA.Matrix<float>.Build.Dense (numHidden, numHidden, 0);
			}

			biasDeltas[numHiddenLayers] = LA.Matrix<float>.Build.Dense (1, numOutputs, 0);
			weightDeltas[numHiddenLayers] = LA.Matrix<float>.Build.Dense (numHidden, numOutputs, 0);

			current_cost = 0f;

			for (int m = 0; m < numTestCases; m++) {
				currInput = (inputs.Row(m)).ToRowMatrix();
				currTarget = (targets.Row(m)).ToRowMatrix();

				w_outputs[w_outputs.Count-1] = ComputeOutputs(currInput);
				ComputeDeltas2(ref currTarget, learningRate, momentum, ref biasDeltas,ref weightDeltas);
				current_cost += CostFunction(currTarget, (LA.Matrix<float>)w_outputs[w_outputs.Count-1]);
			}

			// Weight-adjustment
			for(int l = 0; l <= numHiddenLayers; l++){
				w_biases[l] = ((LA.Matrix<float>)w_biases[l]).Add( ((LA.Matrix<float>)biasDeltas[l]).Divide(numTestCases) );
				w_weights[l] = ((LA.Matrix<float>)w_weights[l]).Add( ((LA.Matrix<float>)weightDeltas[l]).Divide(numTestCases) );
			}

			// Compute new Cost
			current_cost /= (Classification == false)? 2*numTestCases : -numTestCases;
		}
		Debug.Log (numInputs);
		Debug.Log (i);
		Debug.Log (current_cost);
	}
}
