﻿using UnityEngine;
using System.Collections;
using LA = MathNet.Numerics.LinearAlgebra;

public class NetworkController : MonoBehaviour {

	public bool recordMovement;
	public bool debugMovement;
	public bool TrainingPhase;

	private NeuralNetwork[] nets;
	private GameObject target;
	private SupervisedMotion sm;
	private RadialGridSensor sensor;

	// Use this for initialization
	void Start () {
		nets = gameObject.GetComponents<NeuralNetwork> ();
		sm = gameObject.GetComponent<SupervisedMotion>();
		target = GameObject.FindGameObjectWithTag("Goal");
		sensor = gameObject.transform.GetChild(0).gameObject.GetComponent<RadialGridSensor> ();
	}

	private float vectorDiff(Vector3 targetDirection, Vector3 lookDirection){
			
			// the vector perpendicular to referenceForward (90 degrees clockwise)
			// (used to determine if angle is positive or negative)
		Vector3 referenceRight= Vector3.Cross(Vector3.up, lookDirection);
		if ( float.IsNaN(referenceRight.x) 
		    || float.IsNaN(referenceRight.y) 
		    || float.IsNaN(referenceRight.z) )
			return 0f;
			
		// Get the angle in degrees between 0 and 180
		float angle = Vector3.Angle(targetDirection, lookDirection);
		
		// Determine if the degree value should be negative. Here, a positive value
		// from the dot product means that our vector is on the right of the reference vector
		// whereas a negative value means we're on the left.
		float sign = Mathf.Sign(Vector3.Dot(targetDirection, referenceRight));
		
		float finalAngle = sign * angle;
		return finalAngle;
	}

	private void recordSensorData(ref NeuralNetwork net, ArrayList hits, float rotation){
		net.allOutputs.Add(rotation);
		net.allInputs.AddRange(hits);
	}

	private void recordRotation(ref NeuralNetwork net, float rotation, float sensorData){
		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");

		Vector3 goalDirection = target.transform.position - transform.position;
		float angleDiff = vectorDiff(goalDirection, transform.forward); //Vector3.Angle(transform.forward, goalDirection);
		//print ( vectorDiff(goalDirection, transform.forward) );

		//net.numInputs = 1 + hits.Count;
		
		net.allOutputs.Add(rotation);

		net.allInputs.Add(angleDiff);
		net.allInputs.Add(sensorData);
		//net.allInputs.AddRange(hits);
	}

	private void recordDirection(ref NeuralNetwork net, float horizontalMovement, float verticalMovement, float rotAmount){
		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");
		if (sensor == null)
			sensor = gameObject.transform.GetChild(0).gameObject.GetComponent<RadialGridSensor> ();

		//float deltaX = target.transform.position.x - transform.position.x;
		float deltaZ = target.transform.position.z - transform.position.z;
			
		//net.allOutputs.Add(horizontalMovement);
		net.allOutputs.Add (verticalMovement);
		//net.allInputs.Add (deltaX);
		net.allInputs.Add (deltaZ);
		net.allInputs.Add (rotAmount);
	}

	private void trainNetworks(){
		foreach (NeuralNetwork net in nets)
			net.LearningPhase(net.allInputs, net.allOutputs, net.allowedError);
		TrainingPhase = false;
	}

	private Vector3 GetNewDirection(NeuralNetwork net, float rotAmount){
		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");
		LA.Matrix<float> inputs = LA.Matrix<float>.Build.Dense (1, net.numInputs, new float[]{
									//target.transform.position.x - transform.position.x,
									target.transform.position.z - transform.position.z,
									rotAmount});
		LA.Matrix<float> outputs = net.ComputeOutputs(inputs);
		return new Vector3(0, 0, outputs[0,0]);
	}

	private float GetNewSensorData(NeuralNetwork net, ArrayList hits){
		float[] input_array = new float[net.numInputs];
		
		for(int i = 0; i < hits.Count; i++){
			float hit = (float)hits[i];
			input_array[i] = hit; 
		}

		LA.Matrix<float> inputs = LA.Matrix<float>.Build.Dense (1, net.numInputs, input_array);
		
		LA.Matrix<float> outputs = net.ComputeOutputs(inputs);
		return outputs[0,0];
	}

	private float GetNewRotation(NeuralNetwork net, float sensorData){
		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");

		Vector3 goalDirection = target.transform.position - transform.position;
		float angleDiff = vectorDiff(goalDirection, transform.forward); //Vector3.Angle(transform.forward, goalDirection);
		float[] input_array = new float[net.numInputs];
		input_array[0] = angleDiff;
		input_array[1] = sensorData;

		LA.Matrix<float> inputs = LA.Matrix<float>.Build.Dense (1, net.numInputs, input_array);

		LA.Matrix<float> outputs = net.ComputeOutputs(inputs);
		return outputs[0,0];
	}

	// Update is called once per frame
	void FixedUpdate () {
		float verticalMovement = Input.GetAxis ("Vertical");
		float horizontalMovement = Input.GetAxis ("Horizontal");
		float amount = 0f;
		float sensorData = 0f;
		Vector3 direction = new Vector3 (0, 0, verticalMovement);
		ArrayList hits = sensor.collisionCheck();

		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");

		if( recordMovement )
			Debug.DrawLine(transform.position, target.transform.position);

		amount = horizontalMovement * Time.deltaTime * sm.rotateSpeed;

		if( horizontalMovement > 0 )
			sensorData = 1;
		else if( horizontalMovement < 0 )
			sensorData = -1;

		// Handle direction recording
		if ( recordMovement && !TrainingPhase && (direction.x != 0f || direction.z != 0f) )
			recordDirection(ref nets[0], horizontalMovement, verticalMovement, (amount > 0f)? 1:0 );

		// Handle rotation data recording
		if ( recordMovement && !TrainingPhase && (direction.x != 0f || direction.z != 0f || amount != 0f) )
			recordRotation(ref nets[1], amount/2, sensorData);

		// Handle sensor data recording
		if ( recordMovement && !TrainingPhase && (direction.x != 0f || direction.z != 0f || amount != 0f) )
			recordSensorData(ref nets[2], hits, sensorData);

		// Handle Network training
		if ( !recordMovement && TrainingPhase )
			trainNetworks();

		// Obtain new direction
		if ( !debugMovement && !recordMovement && !TrainingPhase ){
			direction = GetNewDirection(nets[0], (amount > 0f)? 1:0 );
			sensorData = GetNewSensorData(nets[2], hits);
			amount = GetNewRotation(nets[1], sensorData) * 2;
		}

		sm.rotate(amount);
		sm.move(direction);
	}
}
