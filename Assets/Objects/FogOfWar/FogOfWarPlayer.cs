using UnityEngine;
using System.Collections;

public class FogOfWarPlayer : MonoBehaviour {

	public Transform FogOfWarPlane;

	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void Update () {
		Vector3 agentPos = new Vector3(transform.position.x, 
		                               Camera.main.nearClipPlane, 
		                               transform.position.z);

		Vector3 screenPos = Camera.main.WorldToScreenPoint (agentPos);
		Ray rayToPlayerPos = Camera.main.ScreenPointToRay (screenPos);

		RaycastHit hit;
		if (Physics.Raycast (rayToPlayerPos, out hit, 1000)) {
			FogOfWarPlane.GetComponent<Renderer>().material.SetVector("_AgentPos", hit.point);
		}
	}
}
