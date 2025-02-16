#%% 3. DEFINE CLASS GRAPH() TO GET DATA FOR SANKEY

""" thanks to the ActivityBrowser people for this, it's much much better than what I had written"""
import bw2calc as bc
import bw2data as bd

class Graph():

    import bw2calc as bc
    import bw2data as bd

    def new_graph(self, data):
        self.json_data = Graph.get_json_data(data)

    @staticmethod
    def get_json_data(data) -> str:
        """Transform bw.Graphtraversal() output to JSON data."""
        lca = data["lca"]
        lca_score = lca.score
        lcia_unit = "x"  # bd.Method(lca.method).metadata["unit"]
        demand = list(lca.demand.items())[0]
        reverse_activity_dict = {v: k for k, v in lca.activity_dict.items()}

        build_json_node = Graph.compose_node_builder(
            lca_score, lcia_unit, demand[0])
        build_json_edge = Graph.compose_edge_builder(
            reverse_activity_dict, lca_score, lcia_unit)

        valid_nodes = (
            (bd.get_activity(reverse_activity_dict[idx]), v)
            for idx, v in data["nodes"].items() if idx != -1
        )
        valid_edges = (
            edge for edge in data["edges"]
            if all(i != -1 for i in (edge["from"], edge["to"]))
        )

        json_data = {
            "nodes": [build_json_node(act, v) for act, v in valid_nodes],
            "edges": [build_json_edge(edge) for edge in valid_edges],
            "title": Graph.build_title(demand, lca_score, lcia_unit),
            "max_impact": max(abs(n["cum"]) for n in data["nodes"].values()),
        }
        #print("JSON DATA (Nodes/Edges):", len(nodes), len(edges))
        # print(json_data)
        return json_data

    @staticmethod
    def build_title(demand: tuple, lca_score: float, lcia_unit: str) -> str:
        act, amount = demand[0], demand[1]
        if type(act) is tuple:
            act = bd.get_activity(act)
        format_str = ("Reference flow: {:.2g} {} {} | {} | {} <br>"
                      "Total impact: {:.2g} {}")
        return format_str.format(
            amount,
            act.get("unit"),
            act.get("reference product") or act.get("name"),
            act.get("name"),
            act.get("location"),
            lca_score, lcia_unit,
        )

    @staticmethod
    def compose_node_builder(lca_score: float, lcia_unit: str, demand: tuple):
        """Build and return a function which processes activities and values
        into valid JSON documents.

        Inspired by https://stackoverflow.com/a/7045809
        """

        def build_json_node(act, values: dict) -> dict:
            return {
                "db": act.key[0],
                "id": act.key[1],
                "product": act.get("reference product") or act.get("name"),
                "name": act.get("name"),
                "location": act.get("location"),
                "amount": values.get("amount"),
                "LCIA_unit": lcia_unit,
                "ind": values.get("ind"),
                "ind_norm": values.get("ind") / lca_score,
                "cum": values.get("cum"),
                "cum_norm": values.get("cum") / lca_score,
                # if act == demand else identify_activity_type(act),
                "class": "demand"
            }

        return build_json_node

    @staticmethod
    def compose_edge_builder(reverse_dict: dict, lca_score: float, lcia_unit: str):
        """Build a function which turns graph edges into valid JSON documents.
        """

        def build_json_edge(edge: dict) -> dict:
            p = bd.get_activity(reverse_dict[edge["from"]])
            from_key = reverse_dict[edge["from"]]
            to_key = reverse_dict[edge["to"]]
            return {
                "source_id": from_key[1],
                "target_id": to_key[1],
                "amount": edge["amount"],
                "product": p.get("reference product") or p.get("name"),
                "impact": edge["impact"],
                "ind_norm": edge["impact"] / lca_score,
                "unit": lcia_unit,
                "tooltip": '<b>{}</b> ({:.2g} {})'
                           '<br>{:.3g} {} ({:.2g}%) '.format(
                    lcia_unit, edge["amount"], p.get("unit"),
                    edge["impact"], lcia_unit, edge["impact"] / lca_score * 100,
                )
            }

        return build_json_edge

    @staticmethod
    def identify_activity_type(act):
        """Return the activity type based on its naming."""
        name = act["name"]
        if "treatment of" in name:
            return "treatment"
        elif "market for" in name:
            # if not "to generic" in name:  # these are not markets, but also transferring activities
            return "market"
        elif "market group" in name:
            # if not "to generic" in name:
            return "marketgroup"
        else:
            return "production"
