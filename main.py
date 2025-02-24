import autogen
from autogen import config_list_from_json
from typing import Dict, List, Any, Optional, TypedDict, Annotated
import json
import datetime


class Transaction(TypedDict):
    transaction_id: str
    sender_account: str
    receiver_account: str
    amount: float
    timestamp: str
    description: Optional[str]
    is_realtime: bool


class FraudAssessment(TypedDict):
    probability: float
    threshold: float
    is_fraud: bool
    features: Dict[str, Any]
    model_version: str


class RuleAssessment(TypedDict):
    is_flagged: bool
    rules_triggered: List[str]
    version: str


class UserHistory(TypedDict):
    account_id: str
    account_age_days: int
    average_transaction_amount: float
    transaction_frequency: float
    previous_flags: int
    typical_countries: List[str]
    typical_receivers: List[str]
    risk_score: float

config_list = [
    {
        "model": "gpt-4o-mini",
        "api_type": "azure",
        "api_version": "2024-06-01",
        "api_key": "api_key",
        "base_url": "base_url"
    }
]

class FraudDetectionSystem:
    def __init__(self):
        # In einer Produktivumgebung würde die Konfiguration aus einer Datei oder Umgebungsvariablen geladen
        self.config_list = config_list
        self.llm_config = {"config_list": self.config_list, "temperature": 0}
        self.llm_config_structured = {
            "config_list": self.config_list,
            "temperature": 0
        }

        # ML-Bewertungsagent
        self.ml_assessment_agent = autogen.AssistantAgent(
            name="ml_assessment_agent",
            system_message="""
            Du bist ein Machine Learning Modell zur Betrugserkennung in Banktransaktionen.
            Du analysierst Transaktionen und bewertest ihre Betrugswahrscheinlichkeit.

            Verwende folgende Faktoren in deiner Bewertung:
            1. Transaktionsbetrag (ungewöhnlich hohe Beträge sind verdächtig)
            2. Empfänger (neue Empfänger sind verdächtiger als bekannte)
            3. Echtzeit-Überweisungen (erhöhtes Risiko)
            4. Zeitpunkt der Transaktion (ungewöhnliche Zeiten sind verdächtig)
            5. Nutzerverhalten im Vergleich zur Historie

            Deine Antwort MUSS eine gültige JSON-Struktur sein und nichts anderes enthalten:
            {
                "probability": 0.75,
                "threshold": 0.5,
                "is_fraud": true,
                "features": {
                    "amount_unusually_high": true,
                    "new_receiver": true,
                    "is_realtime": true,
                    "unusual_time": false
                },
                "model_version": "fraud-detection-v3.2"
            }
            """,
            llm_config=self.llm_config_structured,
        )

        # Regelbasierter Bewertungsagent
        self.rule_assessment_agent = autogen.AssistantAgent(
            name="rule_assessment_agent",
            system_message="""
            Du bist ein regelbasiertes System zur Betrugserkennung in Banktransaktionen.
            Du wendest feste Regeln an, um potenzielle Betrugsfälle zu identifizieren.

            Prüfe folgende Regeln:
            1. Betrag > 5000 EUR -> "large_amount"
            2. Echtzeit-Überweisung -> "realtime_transfer"
            3. Transaktion zwischen 23:00 und 6:00 Uhr -> "unusual_time"
            4. Neue Empfänger-Kontonummer -> "new_receiver"
            5. Ungewöhnliche Beschreibung -> "suspicious_description"

            Deine Antwort MUSS eine gültige JSON-Struktur sein und nichts anderes enthalten:
            {
                "is_flagged": true,
                "rules_triggered": ["large_amount", "realtime_transfer"],
                "version": "rule-engine-v2.1"
            }
            """,
            llm_config=self.llm_config_structured,
        )

        # Erklärungs-Agent
        self.explanation_agent = autogen.AssistantAgent(
            name="explanation_agent",
            system_message="""
            Du bist ein erklärender KI-Agent für ein Betrugsbewertungssystem in einer Bank.
            Deine Aufgabe ist es, die Entscheidungen des Systems in natürlicher Sprache zu erklären.

            Formuliere eine klare, präzise und verständliche Erklärung für den Fraud-Manager, warum eine Transaktion verdächtig erscheint.
            Beziehe dich dabei konkret auf die Bewertungsergebnisse und stelle Zusammenhänge zum Nutzerverhalten her.

            Strukturiere deine Antwort gut und hebe die wichtigsten Faktoren hervor.
            Wenn du Fragen bekommst, nutze die dir zur Verfügung stehenden Tools, um relevante Daten abzurufen.
            """,
            llm_config=self.llm_config,
        )

        # Entscheidungs-Agent
        self.decision_agent = autogen.AssistantAgent(
            name="decision_agent",
            system_message="""
            Du bist ein Entscheidungs-Agent für Echtzeitüberweisungen in einem Bankensystem.
            Du musst autonome Entscheidungen treffen, ob eine Transaktion genehmigt oder abgelehnt werden soll.

            Berücksichtige dabei alle verfügbaren Informationen und wäge das Risiko eines Betrugs gegen die
            Kundenfreundlichkeit ab. Bei Echtzeit-Überweisungen musst du schnell und präzise entscheiden.

            Deine Antwort MUSS eine gültige JSON-Struktur sein und nichts anderes enthalten:
            {
                "decision": "approved",
                "confidence": 0.85,
                "reasoning": "Kurze Begründung deiner Entscheidung"
            }

            Für decision darfst du nur "approved" oder "declined" verwenden.
            """,
            llm_config=self.llm_config_structured,
        )

        # Koordinator-Agent
        self.coordinator_agent = autogen.AssistantAgent(
            name="coordinator_agent",
            system_message="""
            Du bist ein Koordinator-Agent für ein Betrugsbewertungssystem einer Bank.
            Du orchestrierst den Workflow zur Betrugserkennung und steuerst den Prozess zwischen verschiedenen Agenten.

            Deine Aufgaben:
            1. Koordiniere den Informationsfluss zwischen den Bewertungs-Agenten
            2. Entscheide basierend auf den Bewertungen über den weiteren Prozessverlauf
            3. Leite bei Bedarf Informationen an den Fraud-Manager weiter
            4. Stelle bei Echtzeit-Überweisungen eine schnelle Abwicklung sicher

            Treffe klare Entscheidungen über den nächsten Schritt im Prozess.
            """,
            llm_config=self.llm_config,
        )

        # Werkzeuge für den ReAct-Agenten
        def get_user_transaction_history(account_id: Annotated[str, "Die Kontonummer des Nutzers"]) -> str:
            """Ruft die letzten Transaktionen eines Nutzers aus der Datenbank ab."""
            # Simulierte Daten
            transactions = [
                {
                    "transaction_id": "t123456",
                    "amount": 1250.00,
                    "timestamp": "2023-12-01T15:30:00Z",
                    "receiver_account": "DE89370400440532013000",
                    "description": "Monatsmiete Dezember"
                },
                {
                    "transaction_id": "t123457",
                    "amount": 89.99,
                    "timestamp": "2023-12-03T10:15:00Z",
                    "receiver_account": "DE12500105170648489890",
                    "description": "Online-Einkauf Elektronik"
                },
                {
                    "transaction_id": "t123458",
                    "amount": 50.00,
                    "timestamp": "2023-12-05T09:20:00Z",
                    "receiver_account": "DE13600501017832594242",
                    "description": "Überweisung an Freund"
                }
            ]
            return json.dumps(transactions, indent=2)

        def get_user_profile(account_id: Annotated[str, "Die Kontonummer des Nutzers"]) -> str:
            """Ruft das Profil eines Nutzers aus der Datenbank ab."""
            # Simulierte Daten
            user_profile = {
                "account_id": account_id,
                "account_age_days": 730,
                "account_type": "private",
                "risk_score": 0.15,
                "average_transaction_amount": 450.75,
                "transaction_frequency": 12.5,  # Pro Monat
                "previous_flags": 1,
                "typical_countries": ["DE", "FR", "ES"],
                "typical_receivers": ["DE89370400440532013000", "DE12500105170648489890"]
            }
            return json.dumps(user_profile, indent=2)

        def get_similar_fraud_cases(
                case_features: Annotated[str, "JSON-String mit den Features des aktuellen Falls"]) -> str:
            """Findet ähnliche Betrugsfälle basierend auf den gegebenen Merkmalen."""
            # Simulierte Daten
            similar_cases = [
                {
                    "case_id": "f987654",
                    "similarity_score": 0.85,
                    "features": {
                        "amount_unusually_high": True,
                        "new_receiver": True,
                        "unusual_time": True
                    },
                    "outcome": "confirmed_fraud"
                },
                {
                    "case_id": "f987655",
                    "similarity_score": 0.78,
                    "features": {
                        "amount_unusually_high": True,
                        "new_receiver": False,
                        "unusual_time": True
                    },
                    "outcome": "false_positive"
                }
            ]
            return json.dumps(similar_cases, indent=2)

        # ReAct-Agent für Datenbankabfragen
        self.react_agent = autogen.AssistantAgent(
            name="react_agent",
            system_message="""
            Du bist ein erklärender ReAct-Agent für ein Betrugsbewertungssystem einer Bank. 
            Du kannst auf Anfragen des Fraud-Managers reagieren und mithilfe der dir zur Verfügung stehenden Tools zusätzliche 
            Informationen über den Nutzer und seine Transaktionsgeschichte abrufen.

            Nutze die Tools, um relevante Informationen zu sammeln und beantworte die Fragen präzise und hilfreich.
            Strukturiere deine Antworten klar und verwende konkrete Daten, um deine Aussagen zu belegen.

            Die aktuelle Transaktion wurde als potenziell betrügerisch eingestuft, und der Fraud-Manager
            benötigt detaillierte Informationen, um eine fundierte Entscheidung zu treffen.
            """,
            llm_config={
                "config_list": self.config_list,
                "temperature": 0,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_user_transaction_history",
                            "description": "Ruft die letzten Transaktionen eines Nutzers aus der Datenbank ab.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "account_id": {
                                        "type": "string",
                                        "description": "Die Kontonummer des Nutzers"
                                    }
                                },
                                "required": ["account_id"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_user_profile",
                            "description": "Ruft das Profil eines Nutzers aus der Datenbank ab.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "account_id": {
                                        "type": "string",
                                        "description": "Die Kontonummer des Nutzers"
                                    }
                                },
                                "required": ["account_id"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_similar_fraud_cases",
                            "description": "Findet ähnliche Betrugsfälle basierend auf den gegebenen Merkmalen.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "case_features": {
                                        "type": "string",
                                        "description": "JSON-String mit den Features des aktuellen Falls"
                                    }
                                },
                                "required": ["case_features"]
                            }
                        }
                    }
                ]
            }
        )

        # UserProxyAgent für die Interaktion mit dem Fraud-Manager
        self.fraud_manager = autogen.UserProxyAgent(
            name="Fraud_Manager",
            human_input_mode="ALWAYS",  # Fordert immer Benutzereingaben an
            code_execution_config=False,
            system_message="""
            Du bist ein Fraud-Manager in einer Bank. Deine Aufgabe ist es, verdächtige Transaktionen zu überprüfen
            und zu entscheiden, ob sie genehmigt oder abgelehnt werden sollen.

            Du kannst Fragen zu der Transaktion stellen, um mehr Informationen zu erhalten.
            Wenn du genügend Informationen hast, gib eine der folgenden Anweisungen:
            - "GENEHMIGEN" um die Transaktion zu genehmigen
            - "ABLEHNEN" um die Transaktion abzulehnen
            - "BEENDEN" um den Prozess zu beenden
            """,
            is_termination_msg=lambda x: x.get("content", "") and any(
                cmd in x.get("content", "").upper() for cmd in ["BEENDEN", "TERMINATE", "EXIT"])
        )

        # Registrieren der Tool-Funktionen beim UserProxyAgent und React-Agent
        @self.fraud_manager.register_for_execution()
        @self.react_agent.register_for_llm(description="Ruft die Transaktionshistorie eines Nutzers ab")
        def _get_user_transaction_history(account_id: str) -> str:
            return get_user_transaction_history(account_id)

        @self.fraud_manager.register_for_execution()
        @self.react_agent.register_for_llm(description="Ruft das Profil eines Nutzers ab")
        def _get_user_profile(account_id: str) -> str:
            return get_user_profile(account_id)

        @self.fraud_manager.register_for_execution()
        @self.react_agent.register_for_llm(description="Findet ähnliche Betrugsfälle")
        def _get_similar_fraud_cases(case_features: str) -> str:
            return get_similar_fraud_cases(case_features)

    def process_transaction(self, transaction_data: Transaction) -> Dict[str, Any]:
        """
        Verarbeitet eine Transaktion durch das Betrugserkennungssystem.

        Args:
            transaction_data: Die zu prüfende Transaktion

        Returns:
            Ein Dictionary mit dem Ergebnis des Prozesses
        """
        # Workflow initiieren
        chat_results = autogen.initiate_chats(
            [
                # ML-Bewertung und regelbasierte Bewertung parallel starten
                {
                    "sender": self.fraud_manager,
                    "recipient": self.ml_assessment_agent,
                    "message": f"Bewerte folgende Transaktion: {json.dumps(transaction_data, indent=2)}",
                    "clear_history": True,
                    "silent": False,
                    "max_turns": 1,
                    "summary_method": "last_msg",
                },
                {
                    "sender": self.fraud_manager,
                    "recipient": self.rule_assessment_agent,
                    "message": f"Prüfe folgende Transaktion gegen die Regelwerk: {json.dumps(transaction_data, indent=2)}",
                    "clear_history": True,
                    "silent": False,
                    "max_turns": 1,
                    "summary_method": "last_msg",
                }
            ]
        )

        # Extrahiere die Bewertungsergebnisse
        ml_assessment = json.loads(chat_results[0].chat_history[-1]['content'])
        rule_assessment = json.loads(chat_results[1].chat_history[-1]['content'])

        # Koordinator entscheidet über weiteren Verlauf
        coordination_message = f"""
        Koordiniere den weiteren Prozessverlauf basierend auf folgenden Bewertungen:

        Transaktion:
        {json.dumps(transaction_data, indent=2)}

        ML-Bewertung:
        {json.dumps(ml_assessment, indent=2)}

        Regelbasierte Bewertung:
        {json.dumps(rule_assessment, indent=2)}

        Entscheide, welcher der folgenden Schritte als nächstes durchgeführt werden soll:
        1. "generate_explanation" - Bei Verdacht, aber nicht bei Echtzeit-Überweisungen
        2. "decision_agent" - Bei Echtzeit-Überweisungen und Verdacht
        3. "approve_transaction" - Bei keinem Verdacht

        Antworte nur mit einem der drei Befehle ohne weitere Erklärung.
        """

        coordination_result = autogen.initiate_chats(
            [
                {
                    "sender": self.fraud_manager,
                    "recipient": self.coordinator_agent,
                    "message": coordination_message,
                    "clear_history": True,
                    "silent": False,
                    "max_turns": 1,
                }
            ]
        )

        next_step = coordination_result[0].chat_history[-1]['content'].strip()

        # Basierend auf der Koordinator-Entscheidung weiter verfahren
        if next_step == "approve_transaction":
            return {
                "transaction": transaction_data,
                "ml_assessment": ml_assessment,
                "rule_assessment": rule_assessment,
                "final_decision": "approved",
                "explanation": None,
                "process_complete": True
            }

        elif next_step == "decision_agent":
            # Bei Echtzeit-Überweisungen: Automatische Entscheidung
            decision_message = f"""
            Treffe eine automatische Entscheidung für diese Echtzeit-Überweisung:

            Transaktion:
            {json.dumps(transaction_data, indent=2)}

            ML-Bewertung:
            {json.dumps(ml_assessment, indent=2)}

            Regelbasierte Bewertung:
            {json.dumps(rule_assessment, indent=2)}
            """

            decision_result = autogen.initiate_chats(
                [
                    {
                        "sender": self.fraud_manager,
                        "recipient": self.decision_agent,
                        "message": decision_message,
                        "clear_history": True,
                        "silent": False,
                        "max_turns": 1,
                    }
                ]
            )

            decision = json.loads(decision_result[0].chat_history[-1]['content'])

            return {
                "transaction": transaction_data,
                "ml_assessment": ml_assessment,
                "rule_assessment": rule_assessment,
                "final_decision": decision["decision"],
                "explanation": decision["reasoning"],
                "process_complete": True
            }

        elif next_step == "generate_explanation":
            # Erklärung generieren
            explanation_message = f"""
            Erkläre, warum die folgende Transaktion verdächtig erscheint:

            Transaktion:
            {json.dumps(transaction_data, indent=2)}

            ML-Bewertung:
            {json.dumps(ml_assessment, indent=2)}

            Regelbasierte Bewertung:
            {json.dumps(rule_assessment, indent=2)}
            """

            explanation_result = autogen.initiate_chats(
                [
                    {
                        "sender": self.fraud_manager,
                        "recipient": self.explanation_agent,
                        "message": explanation_message,
                        "clear_history": True,
                        "silent": False,
                        "max_turns": 1,
                    }
                ]
            )

            explanation = explanation_result[0].chat_history[-1]['content']

            # Interaktiver Dialog mit dem Fraud-Manager und dem ReAct-Agenten
            initial_message = f"""
            Transaktion zur Überprüfung:
            {json.dumps(transaction_data, indent=2)}

            ML-Bewertung:
            {json.dumps(ml_assessment, indent=2)}

            Regelbasierte Bewertung:
            {json.dumps(rule_assessment, indent=2)}

            Erklärung:
            {explanation}

            Sie können jetzt Fragen zu dieser Transaktion stellen. Der Datenbankabfrage-Agent wird Ihnen helfen,
            weitere Informationen zu bekommen. Wenn Sie bereit sind zu entscheiden, geben Sie "GENEHMIGEN" oder "ABLEHNEN" ein.
            """

            # Interaktiver Chat zwischen Fraud-Manager und ReAct-Agent
            chat_result = autogen.initiate_chats(
                [
                    {
                        "sender": self.fraud_manager,
                        "recipient": self.react_agent,
                        "message": initial_message,
                        "clear_history": False,
                        "silent": False,
                    }
                ]
            )

            # Entscheidung des Fraud-Managers extrahieren
            final_decision = "undecided"
            for message in chat_result[0].chat_history:
                content = message.get('content', '').upper()
                if "GENEHMIGEN" in content:
                    final_decision = "approved"
                    break
                elif "ABLEHNEN" in content:
                    final_decision = "declined"
                    break

            return {
                "transaction": transaction_data,
                "ml_assessment": ml_assessment,
                "rule_assessment": rule_assessment,
                "explanation": explanation,
                "final_decision": final_decision,
                "process_complete": final_decision != "undecided"
            }

        else:
            # Fallback bei unerwarteter Koordinator-Antwort
            return {
                "transaction": transaction_data,
                "ml_assessment": ml_assessment,
                "rule_assessment": rule_assessment,
                "error": f"Unerwartete Koordinator-Antwort: {next_step}",
                "process_complete": False
            }

    def interactive_fraud_manager_session(self, transaction_data: Transaction) -> str:
        """
        Startet eine interaktive Sitzung mit dem Fraud-Manager für eine verdächtige Transaktion.

        Args:
            transaction_data: Die zu überprüfende Transaktion

        Returns:
            Die finale Entscheidung des Fraud-Managers
        """
        print("Bewertung der Transaktion wird durchgeführt...")

        # Automatische Antworten für die ersten Bewertungsschritte
        # (Der eigentliche UserProxyAgent übernimmt erst bei der interaktiven Sitzung)
        auto_proxy = autogen.UserProxyAgent(
            name="Auto_Proxy",
            human_input_mode="ALWAYS",
            code_execution_config=False
        )

        # ML- und Regel-Bewertung durchführen
        ml_result = auto_proxy.initiate_chat(
            self.ml_assessment_agent,
            message=f"Bewerte folgende Transaktion: {json.dumps(transaction_data, indent=2)}"
        )
        rule_result = auto_proxy.initiate_chat(
            self.rule_assessment_agent,
            message=f"Prüfe folgende Transaktion gegen das Regelwerk: {json.dumps(transaction_data, indent=2)}"
        )

        ml_assessment = json.loads(ml_result.chat_history[-1]['content'])
        rule_assessment = json.loads(rule_result.chat_history[-1]['content'])

        # Erklärung generieren
        explanation_result = auto_proxy.initiate_chat(
            self.explanation_agent,
            message=f"""
            Erkläre, warum die folgende Transaktion verdächtig erscheint:

            Transaktion:
            {json.dumps(transaction_data, indent=2)}

            ML-Bewertung:
            {json.dumps(ml_assessment, indent=2)}

            Regelbasierte Bewertung:
            {json.dumps(rule_assessment, indent=2)}
            """
        )

        explanation = explanation_result.chat_history[-1]['content']

        print("Bewertung abgeschlossen. Starte interaktive Überprüfung...")

        # Interaktive Session zwischen Fraud-Manager und ReAct-Agent starten
        initial_message = f"""
        ## Fraud-Alert: Verdächtige Transaktion zur Überprüfung

        ### Transaktionsdetails:
        ```json
        {json.dumps(transaction_data, indent=2)}
        ```

        ### ML-Bewertung:
        ```json
        {json.dumps(ml_assessment, indent=2)}
        ```

        ### Regelbasierte Bewertung:
        ```json
        {json.dumps(rule_assessment, indent=2)}
        ```

        ### Analyse & Erklärung:
        {explanation}

        ---

        Sie können nun Fragen zu dieser Transaktion stellen, um mehr Informationen zu erhalten. 
        Der ReAct-Agent kann die Datenbank abfragen, um Ihnen weitere Einblicke zu geben.

        Beispielfragen:
        - "Wie sieht die Transaktionshistorie dieses Kontos aus?"
        - "Gab es ähnliche Betrugsfälle in der Vergangenheit?"
        - "Zeige mir das Nutzerprofil mit Risikobewertung"

        Wenn Sie bereit sind, eine Entscheidung zu treffen, geben Sie "GENEHMIGEN" oder "ABLEHNEN" ein.
        Wenn Sie den Prozess beenden möchten, geben Sie "BEENDEN" ein.
        """

        # Interaktiver Chat mit dem UserProxyAgent (echter Fraud-Manager)
        self.fraud_manager.initiate_chat(
            self.react_agent,
            message=initial_message
        )

        # Entscheidung des Fraud-Managers extrahieren
        final_decision = "undecided"
        for message in self.fraud_manager.chat_history:
            content = message.get('content', '').upper()
            if "GENEHMIGEN" in content:
                final_decision = "approved"
                break
            elif "ABLEHNEN" in content:
                final_decision = "declined"
                break

        return final_decision


# Beispielnutzung
if __name__ == "__main__":
    import sys

    # Prüfen, ob Autogen-Version mindestens 0.2.0 ist (für initiate_chat-Methode)
    try:
        import pkg_resources

        autogen_version = pkg_resources.get_distribution("pyautogen").version
        version_parts = autogen_version.split('.')
        if int(version_parts[0]) == 0 and int(version_parts[1]) < 2:
            print("HINWEIS: Dieses Skript erfordert pyautogen>=0.2.0")
            print(f"Installierte Version: {autogen_version}")
            print("Bitte aktualisieren Sie mit: pip install -U pyautogen")
            sys.exit(1)
    except:
        print("HINWEIS: Konnte Autogen-Version nicht prüfen. Bei Fehlern bitte pyautogen>=0.2.0 installieren.")

    print("\nFraud Detection System mit UserProxyAgent wird initialisiert...\n")

    # System initialisieren
    fraud_system = FraudDetectionSystem()

    print("System erfolgreich initialisiert!\n")

    # Benutzermenü
    print("=== Fraud Detection System - Hauptmenü ===")
    print("1. Automatische Verarbeitung einer Echtzeit-Überweisung")
    print("2. Interaktive Überprüfung einer verdächtigen Transaktion")
    print("3. Beenden")

    choice = input("\nBitte wählen Sie eine Option (1-3): ")

    if choice == "1":
        # Beispieltransaktion mit Echtzeit-Option
        example_transaction_realtime = {
            "transaction_id": "tx98765",
            "sender_account": "DE55500105173984217489",
            "receiver_account": "FR7630006000011234567890189",
            "amount": 2500.00,
            "timestamp": "2023-12-15T22:45:00Z",
            "description": "Dringende Zahlung",
            "is_realtime": True
        }

        print("\n===== Automatische Entscheidung bei Echtzeit-Überweisung =====")
        print("Transaktion wird automatisch verarbeitet...")
        realtime_result = fraud_system.process_transaction(example_transaction_realtime)
        print(f"Ergebnis: Die Transaktion wurde {realtime_result['final_decision']}.")

    elif choice == "2":
        # Beispieltransaktion für manuelle Überprüfung
        example_transaction_manual = {
            "transaction_id": "tx98766",
            "sender_account": "DE55500105173984217489",
            "receiver_account": "FR7630006000011234567890189",
            "amount": 2500.00,
            "timestamp": "2023-12-15T22:45:00Z",
            "description": "Dringende Zahlung",
            "is_realtime": False  # Nicht-Echtzeit für manuelle Überprüfung
        }

        print("\n===== Interaktive Session mit dem Fraud-Manager =====")
        print("Starte interaktive Session für verdächtige Transaktion...")
        print("Im interaktiven Modus können Sie Fragen zur Transaktion stellen.")
        print("Geben Sie 'GENEHMIGEN', 'ABLEHNEN' oder 'BEENDEN' ein, um die Session abzuschließen.")
        print()

        decision = fraud_system.interactive_fraud_manager_session(example_transaction_manual)

        print()
        print(f"Finale Entscheidung des Fraud-Managers: {decision.upper()}")

    else:
        print("\nProgramm wird beendet.")
        sys.exit(0)