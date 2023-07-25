from client.edge_client import EdgeClient, get_client, run_client


def client_runner(server: str, project: str, data_path: str, epochs: int, untrained: bool, retry_limit: int):
    edge_client: EdgeClient = get_client(
        project=project,
        data_path=data_path,
        epochs=epochs,
        untrained=untrained,
        is_emulated=True,
        no_profile=False,
    )
    run_client(edge_client, server, retry_limit)
