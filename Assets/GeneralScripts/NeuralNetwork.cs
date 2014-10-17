﻿using UnityEngine;
using System.Collections;
using MathNet.Numerics;
using LA = MathNet.Numerics.LinearAlgebra;
using DBL = MathNet.Numerics.LinearAlgebra.Double;

public class NeuralNetwork : MonoBehaviour
{
	// ------------------------------------ Properties ------------------------------------
	public bool TrainingPhase;
	public int numInputs;
	public int numHidden;
	public int numOutputs;
	public int numHiddenLayers;
	private LA.Matrix<float> inputs;
	private LA.Matrix<float> ihWeights;
	private LA.Matrix<float> ihBiases;
	private LA.Matrix<float> ihOutputs;
	private LA.Matrix<float> hoWeights;
	private LA.Matrix<float> hoBiases;
	private LA.Matrix<float> outputs;

	/*private float[,] ih_weights = { {0.1f,0.2f,0.3f,0.4f}, 
						   			{0.5f,0.6f,0.7f,0.8f}, 
						   			{0.9f,1.0f,1.1f,1.2f} };
	private float[,] ho_weights = { {1.3f,1.4f}, 
									{1.5f,1.6f}, 
									{1.7f,1.8f}, 
									{1.9f,2.0f} };
	private float[] ih_biases = { -2.0f, -6.0f, -1.0f, -7.0f };
	private float[] ho_biases = { -2.5f, -5.0f };*/

	// ------------------------------------ Initialization ------------------------------------
	void Start ()
	{
		ihWeights = LA.Matrix<float>.Build.Random (numInputs, numHidden);
		hoWeights = LA.Matrix<float>.Build.Random (numHidden, numOutputs);
		ihBiases = LA.Matrix<float>.Build.Random (1, numHidden);
		hoBiases = LA.Matrix<float>.Build.Random (1, numOutputs);
	}

	public void setInputs(LA.Matrix<float> newInputs){
		LA.Matrix<float> temp_out = ComputeOutputs (newInputs);
	}

	// ------------------------------------ Methods ------------------------------------
	public float sigmoid (float x)
	{
			if (x < -45.0f)
					return 0.0f;
			else if (x > 45.0f)
					return 1.0f;
			else
					return 1 / (1 + Mathf.Exp(-x));
	}
	public float hyperTan (float x)
	{
			if (x < -10.0f) 
					return -1.0f;
			else if (x > 10.0f)
					return 1.0f;
			else
					return (float)Trig.Tanh (x);
	}

	// Feed-Forward Part
	public LA.Matrix<float> ComputeOutputs(LA.Matrix<float> newInputs){
		LA.Matrix<float> ihSums;
		LA.Matrix<float> hoSums;

		inputs = newInputs;
		ihSums = newInputs.Multiply (ihWeights);
		ihOutputs = ihSums.Add (ihBiases);

		for (int i = 0; i < numHidden; i++)
			ihOutputs [0, i] = sigmoid(ihOutputs [0, i]);

		hoSums = ihOutputs.Multiply(hoWeights);
		outputs = hoSums.Add (hoBiases);

		for (int i = 0; i < numOutputs; i++)
			outputs [0, i] = hyperTan(outputs [0, i]);

		return outputs;
	}

	// Back-Propagation Part
	public void UpdateWeights(LA.Matrix<float> targetOutputs, float eta, float alpha){
		LA.Matrix<float> oGrads;
		LA.Matrix<float> hGrads;

		// Compute output gradient
		LA.Matrix<float> hyperTanDerivative = (outputs.Negate()).Add(1);
		hyperTanDerivative = hyperTanDerivative.PointwiseMultiply (outputs.Add (1));
		oGrads = (targetOutputs.Subtract (outputs)).PointwiseMultiply(hyperTanDerivative);

		// Compute hidden gradient
		LA.Matrix<float> grads_weights_sums = hoWeights.TransposeAndMultiply (oGrads);
		LA.Matrix<float> sigmoidDerivative = (ihOutputs.Negate ()).Add (1);
		sigmoidDerivative = ihOutputs.PointwiseMultiply (sigmoidDerivative);
		hGrads = (grads_weights_sums.Transpose()).PointwiseMultiply (sigmoidDerivative);

		// output layer updates
		hoBiases = hoBiases.Add (oGrads.Multiply (eta));

		LA.Matrix<float> outer_sum = ihOutputs.TransposeThisAndMultiply (oGrads);
		hoWeights = hoWeights.Add (outer_sum.Multiply (eta));

		// hidden layer updates
		ihBiases = ihBiases.Add (hGrads.Multiply (eta));

		LA.Matrix<float> hidden_sum = inputs.TransposeThisAndMultiply (hGrads);
		ihWeights = ihWeights.Add (hidden_sum.Multiply (eta));
	}
	// Use this for initialization
	/*
	
	// Update is called once per frame
	void Update () {
		
	}*/
}
