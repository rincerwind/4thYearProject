using UnityEngine;
using System.Collections;
using MathNet.Numerics;
using LA = MathNet.Numerics.LinearAlgebra;
using DBL = MathNet.Numerics.LinearAlgebra.Double;

public class NeuralNetwork : MonoBehaviour
{
	// ------------------------------------ Properties ------------------------------------
	public int numIterations;
	public int numInputs;
	public int numHidden;
	public int numOutputs;
	public int numHiddenLayers;
	public float allowedError;
	public float learningRate;
	public float momentum;
	public ArrayList allOutputs;
	public ArrayList allInputs;
	public int stepFunction = 0; // 0 - hyper tan, 1 - sigmoid, 2 - binary
	public bool Classification = true;

	private int numTestCases;
	private LA.Matrix<float> inputs;
	private LA.Matrix<float> ihWeights;
	private LA.Matrix<float> ihBiases;
	private LA.Matrix<float> ihOutputs;

	private LA.Matrix<float> hoWeights;
	private LA.Matrix<float> hoBiases;
	private LA.Matrix<float> outputs;	// holds the current output of the Net

	private ArrayList w_weights;
	private ArrayList w_biases;
	private ArrayList w_outputs;
	private ArrayList biasDeltas;
	private ArrayList weightDeltas;

	// ------------------------------------ Initialization ------------------------------------
	void Start ()
	{
		ihWeights = LA.Matrix<float>.Build.Random (numInputs, numHidden);
		hoWeights = LA.Matrix<float>.Build.Random (numHidden, numOutputs);
		ihBiases = LA.Matrix<float>.Build.Random (1, numHidden);
		hoBiases = LA.Matrix<float>.Build.Random (1, numOutputs);
		allOutputs = new ArrayList ();
		allInputs = new ArrayList ();

		// ------------------------------ Test Code

		/*w_weights = new ArrayList(numHiddenLayers + 1);
		w_biases = new ArrayList(numHiddenLayers + 1);
		w_outputs = new ArrayList(numHiddenLayers + 1);
		biasDeltas = new ArrayList(numHiddenLayers + 1);
		weightDeltas = new ArrayList(numHiddenLayers + 1);

		initArrayList(ref w_outputs, numHiddenLayers + 1);
		initArrayList(ref biasDeltas, numHiddenLayers + 1);
		initArrayList(ref weightDeltas, numHiddenLayers + 1);
	
		w_weights.Add ( LA.Matrix<float>.Build.Random (numInputs, numHidden) );
		w_biases.Add ( LA.Matrix<float>.Build.Random (1, numHidden) );

		for(int i = 1; i < numHiddenLayers; i++){
			w_weights.Add ( LA.Matrix<float>.Build.Random (numHidden, numHidden) );
			w_biases.Add ( LA.Matrix<float>.Build.Random (1, numHidden) );
		}

		w_weights.Add ( LA.Matrix<float>.Build.Random (numHidden, numOutputs) );
		w_biases.Add ( LA.Matrix<float>.Build.Random (1, numOutputs) );*/

		// ----------------------------------------
	}

	// ------------------------------------ Getters ------------------------------------
	public LA.Matrix<float> GetOutputWeights(){
		return hoWeights;
	}
	
	public LA.Matrix<float> GetHiddenWeights(){
		return ihWeights;
	}

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

	private void sigmoidOnMatrix (LA.Matrix<float> x, int len)
	{
		for (int i = 0; i < len; i++)
			x[0, i] = sigmoid(x[0, i]);
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

	private void hyperTanOnMatrix (LA.Matrix<float> x, int len)
	{
		for (int i = 0; i < len; i++)
			x[0, i] = hyperTan(x[0, i]);
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
		LA.Matrix<float> ihSums;
		LA.Matrix<float> hoSums;
		inputs = newInputs;

		ihSums = inputs.Multiply (ihWeights);
		ihOutputs = ihSums.Add (ihBiases);

		for (int i = 0; i < numHidden; i++)
			ihOutputs [0, i] = sigmoid(ihOutputs [0, i]);

		hoSums = ihOutputs.Multiply(hoWeights);
		outputs = hoSums.Add (hoBiases);

		for (int i = 0; i < numOutputs; i++){
			if( Classification == false )
				outputs [0, i] = hyperTan(outputs [0, i]);
			else
				outputs [0, i] = sigmoid(outputs [0, i]);
		}

		return outputs;
	}

	public LA.Matrix<float> ComputeOutputs2(LA.Matrix<float> newInputs){
		LA.Matrix<float> sums;
		inputs = newInputs;

		sums = inputs.Multiply ((LA.Matrix<float>)w_weights[0]);
		w_outputs[0] = sums.Add ((LA.Matrix<float>)w_biases[0]);
		sigmoidOnMatrix((LA.Matrix<float>)w_outputs[0], numHidden);

		int i = 1;
		for( i = 1; i < w_weights.Count - 1; i++ ){
			sums = ((LA.Matrix<float>)w_outputs[i-1]).Multiply ((LA.Matrix<float>)w_weights[i]);
			w_outputs[i] = sums.Add ((LA.Matrix<float>)w_biases[i]);
			sigmoidOnMatrix((LA.Matrix<float>)w_outputs[i], numHidden);
		}

		sums = ((LA.Matrix<float>)w_outputs[i-1]).Multiply ((LA.Matrix<float>)w_weights[i]);
		w_outputs[i] = sums.Add ((LA.Matrix<float>)w_biases[i]);

		if( Classification == false )
			hyperTanOnMatrix((LA.Matrix<float>)w_outputs[i], numOutputs);
		else
			sigmoidOnMatrix((LA.Matrix<float>)w_outputs[i], numOutputs);
		
		return (LA.Matrix<float>)w_outputs[i];
	}

	// ---------------------------- Back-Propagation Part----------------------------

	// returns the deltas in the different layers
	public ArrayList ComputeDeltas(LA.Matrix<float> targetOutputs, float eta, float alpha){
		LA.Matrix<float> oGrads;
		LA.Matrix<float> hGrads;
		ArrayList deltas = new ArrayList();

		// Compute output gradient
		if( Classification == false ){
			LA.Matrix<float> hyperTanDerivative = (outputs.Negate()).Add(1);
			hyperTanDerivative = hyperTanDerivative.PointwiseMultiply (outputs.Add (1));
			oGrads = (targetOutputs.Subtract (outputs)).PointwiseMultiply(hyperTanDerivative);
		}
		else{
			LA.Matrix<float> sigD = (outputs.Negate ()).Add (1);
			sigD = outputs.PointwiseMultiply (sigD);
			oGrads = (targetOutputs.Subtract (outputs)).PointwiseMultiply(sigD);
		}

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

	public void ComputeDeltas2(LA.Matrix<float> targetOutputs, float eta, float alpha, 
	                               ref ArrayList biasDeltas, 
	                           	   ref ArrayList weightDeltas){
		LA.Matrix<float> grads;
		LA.Matrix<float> sums;
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
			sigDeriv = outputs.PointwiseMultiply (sigDeriv);
			grads = (targetOutputs.Subtract (curr_outputs)).PointwiseMultiply(sigDeriv);
		}

		sums = ((LA.Matrix<float>)w_outputs[w_outputs.Count-2]).TransposeThisAndMultiply(grads);
		biasDeltas[biasDeltas.Count - 1] = 
			((LA.Matrix<float>)biasDeltas[biasDeltas.Count - 1]).Add(grads.Multiply(eta));
		weightDeltas[weightDeltas.Count - 1] = 
			((LA.Matrix<float>)weightDeltas[weightDeltas.Count - 1]).Add(sums.Multiply(eta));

		// Compute hidden-hidden gradients
		for(int i = w_outputs.Count - 2; i > 0; i--){
			print (w_outputs.Count);
			grads_weights_sums = ((LA.Matrix<float>)w_weights[i+1]).TransposeAndMultiply (grads);
			curr_outputs = (LA.Matrix<float>)w_outputs[i];

			sigDeriv = (curr_outputs.Negate ()).Add (1);
			sigDeriv = curr_outputs.PointwiseMultiply (sigDeriv);
			grads = (grads_weights_sums.Transpose()).PointwiseMultiply (sigDeriv);
			sums = ((LA.Matrix<float>)w_outputs[i-1]).TransposeThisAndMultiply(grads);

			biasDeltas[i] = ((LA.Matrix<float>)biasDeltas[i]).Add(grads.Multiply(eta));
			weightDeltas[i] = ((LA.Matrix<float>)weightDeltas[i]).Add(sums.Multiply(eta));
		}
		
		// Compute input-hidden gradient
		grads_weights_sums = ((LA.Matrix<float>)w_weights[1]).TransposeAndMultiply (grads);
		curr_outputs = (LA.Matrix<float>)w_outputs[0];
		
		sigDeriv = (curr_outputs.Negate ()).Add (1);
		sigDeriv = curr_outputs.PointwiseMultiply (sigDeriv);
		grads = (grads_weights_sums.Transpose()).PointwiseMultiply (sigDeriv);
		sums = ((LA.Matrix<float>)inputs).TransposeThisAndMultiply(grads);
		
		biasDeltas[0] = ((LA.Matrix<float>)biasDeltas[0]).Add(grads.Multiply(eta));
		weightDeltas[0] = ((LA.Matrix<float>)weightDeltas[0]).Add(sums.Multiply(eta));
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

		ArrayList deltas; // store hidden/output layer bias deltas and weights deltas

		int i;
		for(i = 1; i <= numIterations && current_cost > target_cost; i++){
			// Acumulate deltas (Needed for weight updates)
			// Given in the same order by UpdateWeights
			LA.Matrix<float> deltaOutputBias = LA.Matrix<float>.Build.Dense (1, numOutputs, 0);
			LA.Matrix<float> deltaOutput = LA.Matrix<float>.Build.Dense (numHidden, numOutputs, 0);
			
			LA.Matrix<float> deltaHiddenBias = LA.Matrix<float>.Build.Dense (1, numHidden, 0);
			LA.Matrix<float> deltaHidden = LA.Matrix<float>.Build.Dense (numInputs, numHidden, 0);

			// ------------------------------ Test Code


			/*biasDeltas[0] = LA.Matrix<float>.Build.Dense (1, numHidden, 0);
			weightDeltas[0] = LA.Matrix<float>.Build.Dense (numInputs, numHidden, 0);

			for(int l = 1; l < numHiddenLayers; l++ ){
				biasDeltas[l] = LA.Matrix<float>.Build.Dense (1, numHidden, 0);
				weightDeltas[l] = LA.Matrix<float>.Build.Dense (numHidden, numHidden, 0);
			}

			biasDeltas[numHiddenLayers] = LA.Matrix<float>.Build.Dense (1, numOutputs, 0);
			weightDeltas[numHiddenLayers] = LA.Matrix<float>.Build.Dense (numHidden, numOutputs, 0);*/

			// ----------------------------------------

			current_cost = 0f;

			for (int m = 0; m < numTestCases; m++) {
				currInput = (inputs.Row(m)).ToRowMatrix();
				currTarget = (targets.Row(m)).ToRowMatrix();
				outputs = ComputeOutputs(currInput);
				//outputs = ComputeOutputs2(currInput);

				deltas = ComputeDeltas(currTarget, learningRate, momentum);
				//ComputeDeltas2(currTarget, learningRate, momentum, ref biasDeltas,ref weightDeltas);

				deltaOutputBias = deltaOutputBias.Add((LA.Matrix<float>)deltas[0]);
				deltaOutput = deltaOutput.Add((LA.Matrix<float>)deltas[1]);
				deltaHiddenBias = deltaHiddenBias.Add((LA.Matrix<float>)deltas[2]);
				deltaHidden = deltaHidden.Add((LA.Matrix<float>)deltas[3]);

				current_cost += CostFunction(currTarget, outputs);
			}
			// UpdateWeights
			hoBiases = hoBiases.Add (deltaOutputBias.Divide(numTestCases));
			hoWeights = hoWeights.Add (deltaOutput.Divide(numTestCases));
			ihBiases = ihBiases.Add (deltaHiddenBias.Divide(numTestCases));
			ihWeights = ihWeights.Add (deltaHidden.Divide(numTestCases));

			// ------------------------------ Test Code

			/*for(int l = 0; l <= numHiddenLayers; l++){
				w_biases[l] = ((LA.Matrix<float>)w_biases[l]).Add( (LA.Matrix<float>)biasDeltas[l] );
				w_weights[l] = ((LA.Matrix<float>)w_weights[l]).Add( (LA.Matrix<float>)weightDeltas[l] );
			}*/

			// ----------------------------------------

			// Compute new Cost
			current_cost /= (Classification == false)? 2*numTestCases : -numTestCases;
		}
		Debug.Log (numInputs);
		Debug.Log (i);
		Debug.Log (current_cost);
	}
}
